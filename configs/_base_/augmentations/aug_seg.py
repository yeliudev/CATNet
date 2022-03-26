# dataset settings
train_pipeline = dict(
    _refine_=True,
    _update_=2,
    img_scale=[(384, 384), (640, 640)],
    multiscale_mode='range')
test_pipeline = dict(
    _update_=1,
    img_scale=[(448, 448), (512, 512), (576, 576)],
    flip=True,
    flip_direction=['horizontal', 'vertical', 'diagonal'])
data = dict(
    train=dict(pipeline=train_pipeline, dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
