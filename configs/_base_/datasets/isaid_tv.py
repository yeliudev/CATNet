# dataset settings
dataset_type = 'ISAIDDataset'
data_root = 'data/isaid/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1400, 800), keep_ratio=True),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1400, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='train/instancesonly_filtered_train.json',
                data_prefix=dict(img='train/images/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='val/instancesonly_filtered_val.json',
                data_prefix=dict(img='val/images/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline)
        ]))
val_dataloader = None
val_evaluator = None
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/instancesonly_filtered_test.json',
        data_prefix=dict(img='test/images/'),
        test_mode=True,
        pipeline=test_pipeline))
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'test/instancesonly_filtered_test.json',
    metric=['bbox', 'segm'],
    classwise=True,
    format_only=True,
    outfile_prefix='work_dirs/isaid_test')
