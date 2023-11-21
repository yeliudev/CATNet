_base_ = 'cat_rcnn_r50_3x_dior.py'
# dataset settings
dataset_type = 'DIORDataset'
data_root = 'data/dior/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1200, 1200), (1200, 1000), (1200, 800), (1200, 600),
                (1200, 400)],
        keep_ratio=True),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackDetInputs')
]
train_dataloader = dict(
    dataset=dict(datasets=[
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
