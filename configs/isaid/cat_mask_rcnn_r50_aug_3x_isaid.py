_base_ = 'cat_mask_rcnn_r50_3x_isaid.py'
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
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
