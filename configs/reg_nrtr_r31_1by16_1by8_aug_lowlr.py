_base_ = [
    '../mmocr/configs/_base_/default_runtime.py', '../mmocr/configs/_base_/recog_models/nrtr.py'
]
load_from = 'nrtr_r31_academic_20210406-954db95e.pth'
label_convertor = dict(type='AttnConvertor', dict_type='DICT36', with_unknown=True, dict_list=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'], lower=False)
model = dict(
    type='NRTR',
    backbone=dict(
        type='ResNet31OCR',
        layers=[1, 2, 5, 3],
        channels=[32, 64, 128, 256, 512, 512],
        stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
        last_stage_pool=True),
    encoder=dict(type='TFEncoder'),
    decoder=dict(type='TFDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=40)

# optimizer
optimizer = dict(type='Adam', lr=1e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[4])
total_epochs = 6

img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomPaddingOCR',
        max_ratio=[0.15, 0.2, 0.15, 0.2],
        box_type=None),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=32,
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
        height=32,
        min_width=32,
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
    samples_per_gpu=800,
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset', datasets=[test], pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset', datasets=[test], pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
