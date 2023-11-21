# dataset settings
dataset_type = 'HRSIDDataset'
data_root = 'data/hrsid/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1200, 1000), keep_ratio=True),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1200, 1000), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations/train2017.json',
            data_prefix=dict(img='images/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/test2017.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline))
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test2017.json',
    metric=['bbox', 'segm'],
    classwise=True)
test_dataloader = val_dataloader
test_evaluator = val_evaluator
