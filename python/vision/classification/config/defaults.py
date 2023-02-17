import os
from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.BASE_MODEL = "efficientnetb0"
_C.MODEL.PRETRAINED = False

_C.DATASETS = CN()
_C.DATASETS.TRAIN = ""
_C.DATASETS.VAL = ""
_C.DATASETS.CLASS_NAMES = []

_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 300
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.CRITERION = "CrossEntropyLoss"
_C.SOLVER.CHECKPOINT_PERIOD = 1  # XXX: Requires to re-set default value
_C.SOLVER.USE_GPU = True

_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 32
_C.DATALOADER.BATCH_SIZE_VAL = 32
# 後方互換性を持たせるためにBATH表記のものも残しておく
_C.DATALOADER.BATH_SIZE = 32
_C.DATALOADER.BATH_SIZE_VAL = 32

_C.INPUT = CN()
_C.INPUT.CENTER_CROP_USE = False
_C.INPUT.CENTER_CROP_SIZE = [224, 224]
_C.INPUT.RANDOM_CROP_USE = False
_C.INPUT.RANDOM_CROP_SIZE = [224, 224]
_C.INPUT.RANDOM_RESIZE_CROP_USE = False
_C.INPUT.RANDOM_RESIZE_CROP_SIZE = [224, 224]
_C.INPUT.RESIZE_USE = True
_C.INPUT.RESIZE_SIZE = [224, 224]
_C.INPUT.PAD_USE = False
_C.INPUT.PAD_PADDING = [4, 4, 4, 4]
_C.INPUT.COLORJITTER_USE = True
_C.INPUT.COLORJITTER_BRIGHTNESS = 0.0
_C.INPUT.COLORJITTER_CONTRAST = 0.0
_C.INPUT.COLORJITTER_SATURATION = 0.0
_C.INPUT.COLORJITTER_HUE = 0.0
_C.INPUT.GRAYSCALE_USE = False
_C.INPUT.GRAYSCALE_CHANNELS = 3
_C.INPUT.RANDOM_GRAYSCALE_USE = False
_C.INPUT.RANDOM_GRAYSCALE_P = 0.1
_C.INPUT.RANDOM_VERTICAL_FLIP_USE = False
_C.INPUT.RANDOM_VERTICAL_FLIP_P = 0.5
_C.INPUT.RANDOM_HORIZONTAL_FLIP_USE = True
_C.INPUT.RANDOM_HORIZONTAL_FLIP_P = 0.5
_C.INPUT.RANDOM_ROTATION_USE = False
_C.INPUT.RANDOM_ROTATION_DEGREES = [-30, 30]
_C.INPUT.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
_C.INPUT.NORMALIZE_STD = [0.229, 0.224, 0.225]

_C.OUTPUT_DIR = "."
_C.TRAIN_VAL_SPLIT_RATE = 0.7
_C.USE_OLD_TORCH = False