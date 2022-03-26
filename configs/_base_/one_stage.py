# dataset settings
train_pipeline = dict(_refine_=True, _update_=5, size_divisor=128)
test_pipeline = dict(_update_=1, transforms=dict(_update_=3, size_divisor=128))
data = dict(
    train=dict(pipeline=train_pipeline, dataset=dict(pipeline=train_pipeline)),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))
# optimizer settings
optimizer = dict(lr=0.005)
