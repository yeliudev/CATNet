# dataset settings
dataset_type = 'DIORDataset'
data_root = 'data/dior/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1200, 800), keep_ratio=True),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1200, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
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
                ann_file='ImageSets/Main/train.txt',
                data_prefix=dict(sub_data_root='./'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline),
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file='ImageSets/Main/val.txt',
                data_prefix=dict(sub_data_root='./'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32),
                pipeline=train_pipeline)
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='ImageSets/Main/test.txt',
        data_prefix=dict(sub_data_root='./'),
        test_mode=True,
        pipeline=test_pipeline))
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='area')
test_dataloader = val_dataloader
test_evaluator = val_evaluator
