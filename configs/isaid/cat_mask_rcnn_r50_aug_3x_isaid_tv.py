_base_ = 'cat_mask_rcnn_r50_3x_isaid_tv.py'
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1400, 1200), (1400, 1000), (1400, 800), (1400, 600),
                (1400, 400)],
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
            type={{_base_.dataset_type}},
            data_root={{_base_.data_root}},
            ann_file='train/instancesonly_filtered_train.json',
            data_prefix=dict(img='train/images/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline),
        dict(
            type={{_base_.dataset_type}},
            data_root={{_base_.data_root}},
            ann_file='val/instancesonly_filtered_val.json',
            data_prefix=dict(img='val/images/'),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
            pipeline=train_pipeline)
    ]))
