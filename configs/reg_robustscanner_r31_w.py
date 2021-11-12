_base_ = [
    '../mmocr/configs/_base_/default_runtime.py',
    '../mmocr/configs/_base_/recog_models/robust_scanner.py'
]

# optimizer
optimizer = dict(type='Adam', lr=1e-3)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
total_epochs = 6
load_from = 'robustscanner_r31_academic-5f05874f.pth'

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=160,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'text', 'valid_ratio'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=48,
        min_width=48,
        max_width=160,
        keep_aspect_ratio=True,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'resize_shape', 'valid_ratio'
        ]),
]

dataset_type = 'OCRDataset'
img_prefix = 'public_training_data/recognizer_data_final_mix'
train_anno_file1 = 'Training Label/train_recognizer_final_mix.txt'
train1 = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=train_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=100,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=False)

test_anno_file1 = 'Training Label/val_recognizer_final_mix.txt'
test = dict(
    type=dataset_type,
    img_prefix=img_prefix,
    ann_file=test_anno_file1,
    loader=dict(
        type='HardDiskLoader',
        repeat=10,
        parser=dict(
            type='LineStrParser',
            keys=['filename', 'text'],
            keys_idx=[0, 1],
            separator=' ')),
    pipeline=None,
    test_mode=True)

data = dict(
    workers_per_gpu=12,
    samples_per_gpu=512,
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset', datasets=[test], pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset', datasets=[test], pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
