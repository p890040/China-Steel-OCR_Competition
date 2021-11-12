_base_ = [
    '../mmocr/configs/_base_/default_runtime.py', '../mmocr/configs/_base_/recog_models/satrn.py'
]

load_from = 'satrn_academic_20211009-cb8b1580.pth'
label_convertor = dict(type='AttnConvertor', dict_type='DICT36', with_unknown=True, dict_list=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'], lower=False)

model = dict(
    type='SATRN',
    backbone=dict(type='ShallowCNN', input_channels=3, hidden_dim=512),
    encoder=dict(
        type='SatrnEncoder',
        n_layers=12,
        n_head=8,
        d_k=512 // 8,
        d_v=512 // 8,
        d_model=512,
        n_position=100,
        d_inner=512 * 4,
        dropout=0.1),
    decoder=dict(
        type='TFDecoder',
        n_layers=6,
        d_embedding=512,
        n_head=8,
        d_model=512,
        d_inner=512 * 4,
        d_k=512 // 8,
        d_v=512 // 8),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=25)

# optimizer
optimizer = dict(type='Adam', lr=3e-4)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[3, 4])
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
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'text', 'valid_ratio',
            'resize_shape'
        ]),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeOCR',
        height=32,
        min_width=100,
        max_width=100,
        keep_aspect_ratio=False,
        width_downsample_ratio=0.25),
    dict(type='ToTensorOCR'),
    dict(type='NormalizeOCR', **img_norm_cfg),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=[
            'filename', 'ori_shape', 'img_shape', 'valid_ratio',
            'resize_shape'
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
    samples_per_gpu=152,
    train=dict(
        type='UniformConcatDataset',
        datasets=[train1],
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset', datasets=[test], pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset', datasets=[test], pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
