# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (csyeliu at comp.polyu.edu.hk)
# -----------------------------------------------------

from mmdet.datasets import DATASETS, CocoDataset


@DATASETS.register_module()
class VHRDataset(CocoDataset):

    CLASSES = ('airplane', 'ship', 'storage tank', 'baseball diamond',
               'tennis court', 'basketball court', 'ground track field',
               'harbor', 'bridge', 'vehicle')
