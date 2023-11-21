_base_ = [
    '../_base_/models/catnet_r50_fpn.py', '../_base_/datasets/dior.py',
    '../_base_/schedules/schedule_3x.py', '../_base_/default_runtime.py'
]
# schedule settings
optim_wrapper = dict(optimizer=dict(lr=0.005))
