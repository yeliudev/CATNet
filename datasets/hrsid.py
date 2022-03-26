# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (csyeliu at comp.polyu.edu.hk)
# -----------------------------------------------------

from mmdet.datasets import DATASETS, CocoDataset


@DATASETS.register_module()
class HRSIDDataset(CocoDataset):

    CLASSES = ('ship', )
