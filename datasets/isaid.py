# -----------------------------------------------------
# Context Aggregation Network
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (coco.ye.liu at connect.polyu.hk)
# -----------------------------------------------------

from mmdet.datasets import Objects365V1Dataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class ISAIDDataset(Objects365V1Dataset):

    METAINFO = {
        'classes':
        ('ship', 'storage_tank', 'baseball_diamond', 'tennis_court',
         'basketball_court', 'Ground_Track_Field', 'Bridge', 'Large_Vehicle',
         'Small_Vehicle', 'Helicopter', 'Swimming_pool', 'Roundabout',
         'Soccer_ball_field', 'plane', 'Harbor')
    }

    # def filter_data(self):
    #     data_info = super(ISAIDDataset, self).filter_data()
    #     return [d for d in data_info if len(d['instances']) <= 1000]
