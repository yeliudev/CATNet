# dataset settings
test_pipeline = dict(
    _update_=1,
    img_scale=[(960, 960), (1024, 1024), (1088, 1088)],
    flip=True,
    flip_direction=['horizontal', 'vertical', 'diagonal'])
data = dict(
    val=dict(pipeline=test_pipeline), test=dict(pipeline=test_pipeline))
