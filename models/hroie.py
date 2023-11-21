# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (coco.ye.liu at connect.polyu.hk)
# -----------------------------------------------------

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models import BaseRoIExtractor
from mmdet.registry import MODELS


@MODELS.register_module()
class HierarchicalRoIExtractor(BaseRoIExtractor):

    def __init__(self,
                 direction,
                 conv_cfg=None,
                 init_cfg=dict(type='Caffe2Xavier', layer='Conv2d'),
                 **kwargs):
        super(HierarchicalRoIExtractor, self).__init__(
            init_cfg=init_cfg, **kwargs)
        assert direction in ('top_down', 'bottom_up')

        self.direction = direction
        self.atts = nn.ModuleList(
            ConvModule(
                self.out_channels * 2,
                self.out_channels,
                1,
                conv_cfg=conv_cfg,
                act_cfg=None) for _ in range(self.num_inputs))

    def forward(self, feats, rois, roi_scale_factor=None):
        # convert fp32 to fp16 when amp is on
        rois = rois.type_as(feats[0])

        assert len(feats) == self.num_inputs

        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *self.roi_layers[0].output_size)

        if rois.size(0) == 0:
            return roi_feats

        # rescale rois (if necessary)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        # crop roi features
        ori_roi_feats = [
            self.roi_layers[i](feat, rois) for i, feat in enumerate(feats)
        ]

        # aggregate roi features
        if self.direction == 'top_down':
            start_idx, end_idx, offset = self.num_inputs - 1, -1, -1
        else:
            start_idx, end_idx, offset = 0, self.num_inputs, 1

        for i in range(start_idx, end_idx, offset):
            hid = torch.cat((roi_feats, ori_roi_feats[i]), dim=1)
            att = self.atts[i](hid).sigmoid()
            roi_feats += ori_roi_feats[i] * att

        return roi_feats
