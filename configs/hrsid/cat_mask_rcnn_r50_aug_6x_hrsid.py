_base_ = 'cat_mask_rcnn_r50_6x_hrsid.py'
# dataset settings
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomChoiceResize',
        scales=[(1200, 1200), (1200, 1100), (1200, 1000), (1200, 900),
                (1200, 800)],
        keep_ratio=True),
    dict(
        type='RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(dataset=dict(pipeline=train_pipeline)))
