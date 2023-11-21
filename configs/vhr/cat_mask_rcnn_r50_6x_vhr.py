_base_ = [
    '../_base_/models/cat_mask_rcnn_r50_fpn.py', '../_base_/datasets/vhr.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=10), mask_head=dict(num_classes=10)))
