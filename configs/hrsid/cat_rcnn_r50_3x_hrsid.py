_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/modules/dfpn.py',
    '../_base_/modules/scp.py', '../_base_/modules/hroie.py',
    '../_base_/datasets/hrsid.py', '../_base_/detection.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
