# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    neck=dict(
        type='DenseFPN',
        stack_times=5,
        norm_cfg=norm_cfg,
        add_extra_convs='_delete_'))
