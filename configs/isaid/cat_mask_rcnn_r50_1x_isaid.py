_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py', '../_base_/modules/dfpn.py',
    '../_base_/modules/scp.py', '../_base_/modules/hroie.py',
    '../_base_/datasets/isaid.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]
