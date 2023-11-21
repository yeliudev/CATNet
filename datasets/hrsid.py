# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (coco.ye.liu at connect.polyu.hk)
# -----------------------------------------------------

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class HRSIDDataset(CocoDataset):

    METAINFO = {'classes': ('ship', )}
