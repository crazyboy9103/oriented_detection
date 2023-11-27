from typing import Tuple, Literal

import torch
from torch import Tensor
from torchvision.ops import (
    batched_nms,
    remove_small_boxes,
    clip_boxes_to_image,
    box_iou,
)

from ops._box_convert import (
    poly2hbb_np,
    poly2obb,
    poly2obb_np,
    obb2poly,
    obb2xyxy,
    hbb2obb,
)
from mmrotate._C import (
    nms_rotated as _C_nms_rotated,
    box_iou_rotated as _C_box_iou_rotated,
)

def remove_small_rotated_boxes(oboxes: Tensor, min_size: float) -> Tensor:
    """
    Remove boxes which contains at least one side smaller than min_size.

    Args:
        boxes (Tensor[N, 5]): boxes in ``(cx, cy, w, h, a)`` format.
        min_size (float): minimum size

    Returns:
        Tensor[K]: indices of the boxes that have both sides
        larger than min_size
    """
    ws, hs = oboxes[:, 2], oboxes[:, 3]
    keep = (ws >= min_size) & (hs >= min_size)
    keep = torch.where(keep)[0]
    return keep

def box_iou_rotated(boxes1: Tensor, boxes2: Tensor, mode_flag: Literal[0, 1] = 0, aligned: bool = False) -> Tensor:
    """ Rotated box IoU. mode_flag and aligned are kept for compatibility with mmrotate implementation.
    Args:
        boxes1, boxes2 (Tensor[N, 5]): boxes in ``(cx, cy, w, h, a)`` format
        mode_flag (int): 0: standard IOU (Union is a+b-a&b), 1: IOU (Union is a)
        aligned (bool): in principle, aligned=True performs better, but the difference is not significant
    
    Returns:
        Tensor[N, N]: the NxN matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    return _C_box_iou_rotated(boxes1, boxes2, mode_flag, aligned)

def nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, multi_label: bool = False):
    """
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.
    
    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """
    if boxes.shape[0] == 0:
        return None
       
    return _C_nms_rotated(boxes, scores, iou_threshold, multi_label)

def batched_nms_rotated(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float
):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 5]):
           boxes where NMS will be performed. They
           are expected to be in (x_ctr, y_ctr, width, height, angle_radians) format
        scores (Tensor[N]):
           scores for each one of the boxes
        idxs (Tensor[N]):
           indices of the categories for each one of the boxes.
        iou_threshold (float):
           discards all overlapping boxes
           with IoU < iou_threshold

    Returns:
        Tensor:
            int64 tensor with the indices of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    assert boxes.shape[-1] == 5

    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    boxes = boxes.float()  # fp16 does not have enough range for batched NMS
    # Strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # Note that batched_nms in torchvision/ops/boxes.py only uses max_coordinate,
    # which won't handle negative coordinates correctly.
    # Here by using min_coordinate we can make sure the negative coordinates are
    # correctly handled.
    max_coordinate = (
        torch.max(boxes[:, 0], boxes[:, 1]) + torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).max()
    min_coordinate = (
        torch.min(boxes[:, 0], boxes[:, 1]) - torch.max(boxes[:, 2], boxes[:, 3]) / 2
    ).min()
    offsets = idxs.to(boxes) * (max_coordinate - min_coordinate + 1)
    boxes_for_nms = boxes.clone()  # avoid modifying the original values in boxes
    boxes_for_nms[:, :2] += offsets[:, None]
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold)
    return keep