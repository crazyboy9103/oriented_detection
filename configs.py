from dataclasses import dataclass
from typing import Optional, Tuple, Literal

@dataclass
class TrainConfig:
    pretrained: bool = True
    pretrained_backbone: bool = True
    progress: bool = True
    num_classes: int = 13 + 1
    trainable_backbone_layers: Literal[0, 1, 2, 3, 4, 5] = 4
    version: Literal[1, 2] = 2
    learning_rate: float = 0.0001

@dataclass
class ModelConfig:
    # transform parameters
    min_size: int = 480
    max_size: int = 640
    image_mean: Optional[Tuple[float, float, float]] = None
    image_std: Optional[Tuple[float, float, float]] = None
    # RPN parameters
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 2000
    rpn_post_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_test: int = 2000
    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    rpn_score_thresh: float = 0.0
    # R-CNN parameters
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 200 # 200
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25
    bbox_reg_weights: Optional[Tuple[float, float, float, float, float]] = (10, 10, 5, 5, 10)

@dataclass
class Kwargs:
    _skip_flip: bool = False
    _skip_image_transform: bool = False
