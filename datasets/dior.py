# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (csyeliu at comp.polyu.edu.hk)
# -----------------------------------------------------

from mmdet.datasets import DATASETS, VOCDataset


@DATASETS.register_module()
class DIORDataset(VOCDataset):

    CLASSES = ('airplane', 'airport', 'baseball field', 'basketball court',
               'bridge', 'chimney', 'dam', 'expressway service area',
               'expressway toll station', 'golf field', 'ground track field',
               'harbor', 'overpass', 'ship', 'stadium', 'storage tank',
               'tennis court', 'train station', 'vehicle', 'wind mill')

    def __init__(self, **kwargs):
        super(VOCDataset, self).__init__(**kwargs)
        self.year = 2012
