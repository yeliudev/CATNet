# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (csyeliu at comp.polyu.edu.hk)
# -----------------------------------------------------

from mmdet.datasets import DATASETS, CocoDataset


@DATASETS.register_module()
class ISAIDDataset(CocoDataset):

    CLASSES = ('ship', 'storage tank', 'baseball diamond', 'tennis court',
               'basketball court', 'ground track field', 'bridge',
               'large vehicle', 'small vehicle', 'helicopter', 'swimming pool',
               'roundabout', 'soccer ball field', 'plane', 'harbor')
