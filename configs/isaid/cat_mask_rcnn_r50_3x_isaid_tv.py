_base_ = [
    '../_base_/models/cat_mask_rcnn_r50_fpn.py',
    '../_base_/datasets/isaid_tv.py', '../_base_/schedules/schedule_3x.py',
    '../_base_/default_runtime.py'
]
# schedule settings
val_cfg = None
