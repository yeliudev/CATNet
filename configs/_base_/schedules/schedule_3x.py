_base_ = 'mmdet::_base_/schedules/schedule_1x.py'
# schedule settings
train_cfg = dict(max_epochs=36)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=36,
        by_epoch=True,
        milestones=[28, 34],
        gamma=0.1)
]
optim_wrapper = dict(optimizer=dict(lr=0.01))
auto_scale_lr = dict(base_batch_size=8)
