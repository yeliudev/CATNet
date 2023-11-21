# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (coco.ye.liu at connect.polyu.hk)
# -----------------------------------------------------

from mmdet.datasets import XMLDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class DIORDataset(XMLDataset):

    METAINFO = {
        'classes':
        ('airplane', 'airport', 'baseball field', 'basketball court', 'bridge',
         'chimney', 'dam', 'expressway service area',
         'expressway toll station', 'golf field', 'ground track field',
         'harbor', 'overpass', 'ship', 'stadium', 'storage tank',
         'tennis court', 'train station', 'vehicle', 'wind mill')
    }
