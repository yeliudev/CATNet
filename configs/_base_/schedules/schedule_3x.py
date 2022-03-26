_base_ = 'schedule_1x.py'
# learning policy settings
lr_config = dict(step=[28, 34])
runner = dict(max_epochs=36)
