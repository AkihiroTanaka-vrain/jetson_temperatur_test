from .config import *

def make_custom_config(y_cfg, name):
    if y_cfg.USE_ORIGINAL_COCO:
        label_map = COCO_LABEL_MAP
        class_names = COCO_CLASSES
    else:
        label_map = {i:i for i in range(1, len(y_cfg.DATASET.CLASS_NAMES)+1)}
        class_names = y_cfg.DATASET.CLASS_NAMES

    custom_dataset = dataset_base.copy({
        'name': name,
        
        'train_info': y_cfg.DATASET.TRAIN_JSON,
        'train_images': y_cfg.DATASET.TRAIN_IMAGES,
        'valid_info': y_cfg.DATASET.VAL_JSON,
        'valid_images': y_cfg.DATASET.VAL_IMAGES,
        'label_map': label_map,
        'class_names': class_names
    })
    
    custom_base_config = coco_base_config.copy({
        'name': name,

        # Dataset stuff
        'dataset': custom_dataset,
        'num_classes': len(custom_dataset.class_names) + 1,

        # Image Size
        'max_size': 550,
        
        # Training params
        'lr_steps': (280000, 600000, 700000, 750000),
        'max_iter': 800000,
        
        # Backbone Settings
        'backbone': resnet50_backbone.copy({
            'selected_layers': list(range(1, 4)),
            
            'pred_scales': yolact_base_config.backbone.pred_scales,
            'pred_aspect_ratios': yolact_base_config.backbone.pred_aspect_ratios,
            'use_pixel_scales': True,
            'preapply_sqrt': False,
            'use_square_anchors': True, # This is for backward compatability with a bug
        }),
        # FPN Settings
        'fpn': fpn_base.copy({
            'use_conv_downsample': True,
            'num_downsample': 2,
        }),

        # Mask Settings
        'mask_type': mask_type.lincomb,
        'mask_alpha': 6.125,
        'mask_proto_src': 0,
        'mask_proto_net': [(256, 3, {'padding': 1})] * 3 + [(None, -2, {}), (256, 3, {'padding': 1})] + [(32, 1, {})],
        'mask_proto_normalize_emulate_roi_pooling': True,

        # Other stuff
        'share_prediction_module': True,
        'extra_head_net': [(256, 3, {'padding': 1})],

        'positive_iou_threshold': 0.5,
        'negative_iou_threshold': 0.4,

        'crowd_iou_threshold': 0.7,

        'use_semantic_segmentation_loss': True,

        # SSD data augmentation parameters
        # Randomize hue, vibrance, etc.
        'augment_photometric_distort': y_cfg.INPUT.FLAG.PHOTOMETRICDISTORT,
        # Have a chance to scale down the image and pad (to emulate smaller detections)
        'augment_expand': y_cfg.INPUT.FLAG.PAD,
        # Potentialy sample a random crop from the image and put it in a random place
        'augment_random_sample_crop': y_cfg.INPUT.FLAG.CROP,
        # Mirror the image with a probability of 1/2
        'augment_random_mirror': y_cfg.INPUT.FLAG.RHF,
        # Flip the image vertically with a probability of 1/2
        'augment_random_flip': y_cfg.INPUT.FLAG.RVF,
        # With uniform probability, rotate the image [0,90,180,270] degrees
        'augment_random_rot90': y_cfg.INPUT.FLAG.ROT90,
    })

    return custom_base_config
