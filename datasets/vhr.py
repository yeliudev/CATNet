# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (coco.ye.liu at connect.polyu.hk)
# -----------------------------------------------------

from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class VHRDataset(CocoDataset):

    METAINFO = {
        'classes': ('airplane', 'ship', 'storage tank', 'baseball diamond',
                    'tennis court', 'basketball court', 'ground track field',
                    'harbor', 'bridge', 'vehicle')
    }
