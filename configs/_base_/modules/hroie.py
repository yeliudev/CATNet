# model settings
model = dict(
    roi_head=dict(
        bbox_roi_extractor=dict(
            _refine_=True, type='HRoIE', direction='bottom_up'),
        mask_roi_extractor=dict(
            _refine_=True, type='HRoIE', direction='top_down')))
