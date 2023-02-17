import dataclasses
from typing import List

from vision.baseclasses.config import Config


@dataclasses.dataclass
class ModelConfig(Config):
    BASE_MODEL: str = "efficientnetb0"
    PRETRAINED: bool = False


@dataclasses.dataclass
class DatasetsConfig(Config):
    TRAIN: str = ""
    VAL: str = ""
    CLASS_NAMES: List[str] = dataclasses.field(default_factory=lambda: [])


@dataclasses.dataclass
class SolverConfig(Config):
    MAX_EPOCH: int = 300
    BASE_LR: float = 0.001
    BIAS_LR_FACTOR: float = 2
    MOMENTUM: float = 0.9
    WEIGHT_DECAY: float = 0.0005
    WEIGHT_DECAY_BIAS: float = 0.0005
    CRITERION: str = 'CrossEntropyLoss'
    CHECKPOINT_PERIOD: int = 1
    USE_GPU: bool = True


@dataclasses.dataclass
class DataloaderConfig(Config):
    BATCH_SIZE: int = 32
    BATCH_SIZE_VAL: int = 32
    BATH_SIZE: int = 32  # deprecated
    BATH_SIZE_VAL: int = 32  # deprecated


@dataclasses.dataclass
class InputConfig(Config):
    CENTER_CROP_USE: bool = False
    CENTER_CROP_SIZE: List[int] = dataclasses.field(default_factory=lambda: [224, 224])
    RANDOM_CROP_USE: bool = False
    RANDOM_CROP_SIZE: List[int] = dataclasses.field(default_factory=lambda: [224, 224])
    RANDOM_RESIZE_CROP_USE: bool = False
    RANDOM_RESIZE_CROP_SIZE: List[int] = dataclasses.field(default_factory=lambda: [224, 224])
    RESIZE_USE: bool = True
    RESIZE_SIZE: List[int] = dataclasses.field(default_factory=lambda: [224, 224])
    PAD_USE: bool = False
    PAD_PADDING: List[int] = dataclasses.field(default_factory=lambda: [4, 4, 4, 4])
    COLORJITTER_USE: bool = True
    COLORJITTER_BRIGHTNESS: float = 0.0
    COLORJITTER_CONTRAST: float = 0.0
    COLORJITTER_SATURATION: float = 0.0
    COLORJITTER_HUE: float = 0.0
    GRAYSCALE_USE: bool = False
    GRAYSCALE_CHANNELS: int = 3
    RANDOM_GRAYSCALE_USE: bool = False
    RANDOM_GRAYSCALE_P: float = 0.1
    RANDOM_VERTICAL_FLIP_USE: bool = False
    RANDOM_VERTICAL_FLIP_P: float = 0.5
    RANDOM_HORIZONTAL_FLIP_USE: bool = True
    RANDOM_HORIZONTAL_FLIP_P: float = 0.5
    RANDOM_ROTATION_USE: bool = False
    RANDOM_ROTATION_DEGREES: List[float] = dataclasses.field(default_factory=lambda: [-30.0, 30.0])
    NORMALIZE_MEAN: List[float] = dataclasses.field(default_factory=lambda: [0.485, 0.456, 0.406])
    NORMALIZE_STD: List[float] = dataclasses.field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclasses.dataclass
class ClassificationConfig(Config):
    MODEL: ModelConfig = dataclasses.field(default_factory=lambda: ModelConfig())
    DATASETS: DatasetsConfig = dataclasses.field(default_factory=lambda: DatasetsConfig())
    SOLVER: SolverConfig = dataclasses.field(default_factory=lambda: SolverConfig())
    DATALOADER: DataloaderConfig = dataclasses.field(default_factory=lambda: DataloaderConfig())
    INPUT: InputConfig = dataclasses.field(default_factory=lambda: InputConfig())
    OUTPUT_DIR: str = "."
    TRAIN_VAL_SPLIT_RATE: float = 0.7
    USE_OLD_TORCH: bool = False
