# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (csyeliu at comp.polyu.edu.hk)
# -----------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, caffe2_xavier_init, constant_init
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models import NECKS


class ContextAggregation(nn.Module):
    """
    Context Aggregation Block.

    Args:
        in_channels (int): Number of input channels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
    """

    def __init__(self, in_channels, reduction=1, conv_cfg=None):
        super(ContextAggregation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.inter_channels = max(in_channels // reduction, 1)

        conv_params = dict(kernel_size=1, conv_cfg=conv_cfg, act_cfg=None)

        self.a = ConvModule(in_channels, 1, **conv_params)
        self.k = ConvModule(in_channels, 1, **conv_params)
        self.v = ConvModule(in_channels, self.inter_channels, **conv_params)
        self.m = ConvModule(self.inter_channels, in_channels, **conv_params)

        self.init_weights()

    def init_weights(self):
        for m in (self.a, self.k, self.v):
            caffe2_xavier_init(m.conv)
        constant_init(self.m.conv, 0)

    def forward(self, x):
        n, c = x.size(0), self.inter_channels

        # a: [N, 1, H, W]
        a = self.a(x).sigmoid()

        # k: [N, 1, HW, 1]
        k = self.k(x).view(n, 1, -1, 1).softmax(2)

        # v: [N, 1, C, HW]
        v = self.v(x).view(n, 1, c, -1)

        # y: [N, C, 1, 1]
        y = torch.matmul(v, k).view(n, c, 1, 1)
        y = self.m(y) * a

        return x + y


@NECKS.register_module()
class SCP(BaseModule):
    """
    Spatial Context Pyramid.

    Args:
        in_channels (int): Number of input channels.
        num_levels (int): Number of feature pyramid levels.
        reduction (int, optional): Channel reduction ratio. Default: 1.
        conv_cfg (dict or None, optional): Config dict for the convolution
            layer. Default: None.
        init_cfg (dict or list[dict] or None, optional): Initialization config
            dict. Default: None.
    """

    def __init__(self,
                 in_channels,
                 num_levels,
                 reduction=1,
                 conv_cfg=None,
                 init_cfg=None):
        super(SCP, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.num_levels = num_levels

        self.blocks = nn.ModuleList(
            ContextAggregation(
                in_channels, reduction=reduction, conv_cfg=conv_cfg)
            for _ in range(num_levels))

    @auto_fp16()
    def forward(self, inputs):
        out = []
        for i in range(self.num_levels):
            out.append(self.blocks[i](inputs[i]))
        return tuple(out)
