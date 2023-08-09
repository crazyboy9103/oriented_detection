from typing import Tuple

import torch
from torch import Tensor
from torchvision.ops import (
    nms, 
    batched_nms,
    remove_small_boxes,
    clip_boxes_to_image,
    box_convert,
    box_area,
    box_iou,
    generalized_box_iou,
    complete_box_iou,
    distance_box_iou,
    masks_to_boxes
)

from _box_convert import (
    rbbox2result,
    rbbox2roi,
    poly2obb,
    poly2obb_np,
    obb2hbb,
    obb2poly,
    obb2poly_np,
    obb2xyxy,
    hbb2obb,
    norm_angle
)
from torch.ops.detectron2 import (
    nms_rotated,
    box_iou_rotated,
    roi_align_rotated_forward,
    roi_align_rotated_backward
)