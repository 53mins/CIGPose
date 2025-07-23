_base_ = ['mmpose::_base_/default_runtime.py']

# common setting
num_keypoints = 133
input_size = (288, 384)

# runtime
max_epochs = 420
stage2_num_epochs = 150
base_lr = 2e-3
train_batch_size = 32
val_batch_size = 32

train_cfg = dict(max_epochs=max_epochs, val_interval=10)
randomness = dict(seed=21)

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True
)

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

skeleton = [(0, 1), (0, 2), (1, 3), (2, 4), (1, 2), (3, 5), (4, 6), (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (0, 3), (0, 4), (0, 5), (0, 6), (16, 20), (16, 21), (16, 22), (20, 22), (21, 22), (17, 19), (18, 19), (15, 17), (15, 18), (15, 19), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (40, 41), (41, 42), (42, 43), (43, 44), (45, 46), (46, 47), (47, 48), (48, 49), (50, 51), (51, 52), (52, 53), (54, 55), (55, 56), (56, 57), (57, 58), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 59), (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (70, 65), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82), (82, 71), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), (90, 83), (50, 62), (50, 65), (53, 54), (53, 58), (53, 56), (44, 62), (44, 50), (45, 50), (45, 65), (54, 74), (74, 58), (56, 74), (71, 74), (74, 77), (80, 77), (80, 71), (71, 83), (87, 77), (0, 53), (0, 52), (0, 51), (0, 50), (1, 65), (1, 66), (1, 67), (1, 68), (1, 69), (1, 70), (2, 59), (2, 60), (2, 61), (2, 62), (2, 63), (2, 64), (3, 39), (3, 38), (3, 37), (4, 23), (4, 24), (4, 25), (91, 92), (92, 93), (93, 94), (94, 95), (96, 97), (97, 98), (98, 99), (100, 101), (101, 102), (102, 103), (104, 105), (105, 106), (106, 107), (108, 109), (109, 110), (110, 111), (91, 96), (91, 100), (91, 104), (91, 108), (112, 113), (113, 114), (114, 115), (115, 116), (117, 118), (118, 119), (119, 120), (121, 122), (122, 123), (123, 124), (125, 126), (126, 127), (127, 128), (129, 130), (130, 131), (131, 132), (112, 117), (112, 121), (112, 125), (112, 129), (9, 91), (9, 92), (10, 112), (10, 113)]

keypoint_groups_definition = [
    # ---- COCO keypoints ----
    list(range(0, 5)),    # Face
    [5, 7, 9],            # Right shoulder, elbow, wrist
    [6, 8, 10],           # Left shoulder, elbow, wrist
    [11, 12],             # Hips
    [11, 13, 15],         # Right hip, knee, ankle
    [12, 14, 16],         # Left hip, knee, ankle
    # ---- COCO wholebody keypoints ----
    # Face
    list(range(23, 40)),  # Face contour points
    list(range(40, 45)),  # Left eyebrow contour points
    list(range(45, 50)),  # Right eyebrow contour points
    list(range(50, 59)),  # Nose points
    list(range(59, 65)),  # Left eye contour points
    list(range(65, 71)),  # Right eye contour points
    list(range(71, 91)),  # Mouth contour points
    list(range(23, 91)),  # All face points
    # Right hand
    list(range(91, 112)), # Right hand (21 points)
    list(range(92, 96)),  # Right hand thumb
    list(range(96, 100)), # Right hand index finger
    list(range(100, 104)),# Right hand middle finger
    list(range(104, 108)),# Right hand ring finger
    list(range(108, 112)),# Right hand little finger
    [9, 91],              # Right hand wrist
    # Left hand
    list(range(112, 133)),# Left hand (21 points)
    list(range(113, 117)),# Left hand thumb
    list(range(117, 121)),# Left hand index finger
    list(range(121, 125)),# Left hand middle finger
    list(range(125, 129)),# Left hand ring finger
    list(range(129, 133)),# Left hand little finger
    [10, 112],            # Left hand wrist
    # Feet
    list(range(17, 20)),  # Right foot
    list(range(20, 23)),  # Left foot
    [15, 17, 18, 19],     # Right foot
    [16, 20, 21, 22],     # Left foot
]

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=135,
        end=270,
        T_max=135,
        by_epoch=True,
        convert_to_iter_based=True),
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(
    type='SimCCLabel',
    input_size=input_size,
    sigma=(6., 6.93),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=1.,
        widen_factor=1.,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/'
            'rtmposev1/cspnext-l_udp-aic-coco_210e-256x192-273b7631_20230130.pth'  # noqa
        )),
    head=dict(
        type='CIGHead',
        in_channels=1024,
        out_channels=num_keypoints,
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        in_featuremap_num=1,
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=512,
            s=256,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            # type='AdaKLDiscretLoss_v1',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        loss_cf=dict(
            type='CounterfactualConsistencyLoss',
            use_kl=True,
            loss_weight=0.1,
        ),
        decoder=codec,
        gcn_cfg=dict(
            num_layers=1,
            input_dim=512,
            hidden_dim=512,
        ),
        cim_cfg=dict(
            intervention_strategy='topk', # 'topk' or 'threshold'
            intervention_k=13,
            intervention_k_val=1, 
        ),
        keypoint_connections=skeleton,
        keypoint_groups=keypoint_groups_definition,
        ),
    test_cfg=dict(flip_test=True, ))

# base dataset settings
dataset_type = 'UBody2dDataset'
data_mode = 'topdown'
data_root = '../data/UBody/'

backend_args = dict(backend='local')

scenes = [
    'Magic_show', 'Entertainment', 'ConductMusic', 'Online_class', 'TalkShow',
    'Speech', 'Fitness', 'Interview', 'Olympic', 'TVShow', 'Singing',
    'SignLanguage', 'Movie', 'LiveVlog', 'VideoConference'
]

train_datasets = [
    dict(
        type='CocoWholeBodyDataset',
        data_root='../data/coco/',
        data_mode=data_mode,
        ann_file='annotations/coco_wholebody_train_v1.0.json',
        data_prefix=dict(img='train2017/'),
        pipeline=[])
]

for scene in scenes:
    train_dataset = dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file=f'annotations/{scene}/train_annotations.json',
        data_prefix=dict(img='images/'),
        pipeline=[],
        sample_interval=10)
    train_datasets.append(train_dataset)

# pipelines
train_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform', scale_factor=[0.5, 1.5], rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=backend_args),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.5, 1.5],
        rotate_factor=90),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    batch_size=train_batch_size,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco_wholebody.py'),
        datasets=train_datasets,
        pipeline=train_pipeline,
        test_mode=False,
    ))

val_dataloader = dict(
    batch_size=val_batch_size,
    num_workers=10,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoWholeBodyDataset',
        data_root='../data',
        data_mode=data_mode,
        ann_file='coco/annotations/coco_wholebody_val_v1.0.json',
        bbox_file='../data/coco/person_detection_results/COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(
        save_best='coco-wholebody/AP', rule='greater', max_keep_ckpts=1))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# evaluators
val_evaluator = dict(
    type='CocoWholeBodyMetric',
    ann_file='../data/coco/annotations/coco_wholebody_val_v1.0.json')
test_evaluator = val_evaluator

# visualizer = dict(vis_backends=[dict(type='LocalVisBackend'),
#                                 dict(type='TensorboardVisBackend'),
#                                 dict(type='WandbVisBackend'),
#                                 ])