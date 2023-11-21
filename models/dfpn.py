# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (coco.ye.liu at connect.polyu.hk)
# -----------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class DenseFPN(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack_times=1,
                 reduction=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(type='Caffe2Xavier', layer='Conv2d')):
        super(DenseFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        assert isinstance(stack_times, int) and stack_times > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.stack_times = stack_times
        self.reduction = reduction
        self.inter_channels = max(out_channels // reduction, 1)

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level

        # build lateral convs
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            lateral_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)

        # add extra convs on laterals (if necessary)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        self.with_extra_convs = extra_levels >= 1
        if self.with_extra_convs:
            self.extra_convs = nn.ModuleList()
            for _ in range(extra_levels):
                extra_conv = ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.extra_convs.append(extra_conv)

        # build top-down and bottom-up paths
        self.stages = nn.ModuleList()
        for _ in range(stack_times):
            stage = nn.ModuleList(
                nn.Sequential(
                    ConvModule(
                        out_channels,
                        self.inter_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    ConvModule(
                        self.inter_channels,
                        self.inter_channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg),
                    ConvModule(
                        self.inter_channels,
                        out_channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg)) for _ in range(num_outs * 2))
            self.stages.append(stage)

        # build attention coefficients
        self.top_down_atts = nn.ParameterList(
            nn.Parameter(torch.ones(i)) for i in range(num_outs - 1, 1, -1))
        self.bottom_up_atts = nn.ParameterList(
            nn.Parameter(torch.ones(i)) for i in range(2, num_outs))

    def forward(self, inputs):
        # lateral convs
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # extra convs on laterals
        if self.with_extra_convs:
            for i, extra_conv in enumerate(self.extra_convs):
                laterals.append(extra_conv(laterals[-1]))

        for stage in self.stages:
            # top-down path
            top_downs = [None] * self.num_outs
            for i in range(self.num_outs - 1, -1, -1):
                atts = self.top_down_atts[i].softmax(
                    0) if i < self.num_outs - 2 else [1]
                fused = laterals[i] + sum(atts[j] * F.interpolate(
                    laterals[self.num_outs - j - 1],
                    size=laterals[i].shape[2:],
                    mode='bilinear',
                    align_corners=False) for j in range(self.num_outs - i - 1))
                top_downs[i] = stage[i](F.relu(fused))

            # bottom-up path
            bottom_ups = []
            for i in range(self.num_outs):
                atts = self.bottom_up_atts[i - 2].softmax(0) if i > 1 else [1]
                fused = laterals[i] + top_downs[i] + sum(
                    atts[j] *
                    F.adaptive_max_pool2d(top_downs[j], laterals[i].shape[2:])
                    for j in range(i))
                bottom_ups.append(stage[i + self.num_outs](F.relu(fused)))

            laterals = bottom_ups

        return tuple(laterals)
