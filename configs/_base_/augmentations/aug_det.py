# dataset settings
train_pipeline = dict(
    _refine_=True,
    _update_=2,
    img_scale=[(640, 640), (960, 960)],
    multiscale_mode='range')
test_pipeline = dict(
    _update_=1,
    img_scale=[(732, 732), (800, 800), (864, 864)],
    flip=True,
    flip_direction=['horizontal', 'vertical', 'diagonal'])
data = dict(
    train=dict(pipeline=train_pipeline, dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
