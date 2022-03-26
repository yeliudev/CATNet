# dataset settings
train_pipeline = dict(
    _refine_=True,
    _update_=dict(
        index=[1, -1],
        value=[
            dict(with_mask='_delete_'),
            dict(keys=['img', 'gt_bboxes', 'gt_labels'])
        ]))
data = dict(
    train=dict(pipeline=train_pipeline, dataset=dict(pipeline=train_pipeline)))
# evaluation settings
evaluation = dict(metric='bbox')
