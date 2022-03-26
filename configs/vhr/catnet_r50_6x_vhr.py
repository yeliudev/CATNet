_base_ = [
    '../_base_/models/retinanet_r50_fpn.py', '../_base_/modules/dfpn.py',
    '../_base_/modules/scp.py', '../_base_/datasets/vhr.py',
    '../_base_/one_stage.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
model = dict(bbox_head=dict(num_classes=10))
