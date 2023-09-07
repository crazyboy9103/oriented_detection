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
    obb2poly_np,
    obb2xyxy,
    hbb2obb,
)
from mmrotate._C import (
    nms_rotated as _C_nms_rotated,
    box_iou_rotated as _C_box_iou_rotated,
)

# TODO 
def clip_rotated_boxes_to_image(oboxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    Clip boxes so that they lie inside an image of size `size`.

    Args:
        boxes (Tensor[N, 5]): boxes in ``(cx, cy, w, h, a)`` format.
        size (Tuple[height, width]): size of the image

    Returns:
        Tensor[N, 5]: clipped boxes
    """
    # TODO jit
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(clip_boxes_to_image)
    dim = oboxes.dim()
    boxes_cx = oboxes[..., 0:1]
    boxes_cy = oboxes[..., 1:2]
    height, width = size
    
    # TODO jit
    # if torchvision._is_tracing():
    #     boxes_x = torch.max(boxes_x, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
    #     boxes_x = torch.min(boxes_x, torch.tensor(width, dtype=boxes.dtype, device=boxes.device))
    #     boxes_y = torch.max(boxes_y, torch.tensor(0, dtype=boxes.dtype, device=boxes.device))
    #     boxes_y = torch.min(boxes_y, torch.tensor(height, dtype=boxes.dtype, device=boxes.device))
    # else:
    boxes_cx = boxes_cx.clamp(min=0, max=width)
    boxes_cy = boxes_cy.clamp(min=0, max=height)

    clipped_boxes = torch.stack((boxes_cx, boxes_cy), dim=dim)
    return clipped_boxes.reshape(oboxes.shape)

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
    # TODO jit
    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(remove_small_boxes)
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

def nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
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
    
    multi_label = False
    return _C_nms_rotated(boxes, scores, iou_threshold, multi_label)

@torch.jit.script_if_tracing
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
           are expected to be in (x_ctr, y_ctr, width, height, angle_degrees) format
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

# TODO 
def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms_iou_threshold,
                           score_factors=None,):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.

    Returns:
        tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
        (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        return inds

    # Strictly, the maximum coordinates of the rotating box (x,y,w,h,a)
    # should be calculated by polygon coordinates.
    # But the conversion from rbbox to polygon will slow down the speed.
    # So we use max(x,y) + max(w,h) as max coordinate
    # which is larger than polygon max coordinate
    # max(x1, y1, x2, y2,x3, y3, x4, y4)
    max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    if bboxes.size(-1) == 5:
        bboxes_for_nms = bboxes.clone()
        bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    else:
        bboxes_for_nms = bboxes + offsets[:, None]
        
    keep = nms_rotated(bboxes_for_nms, scores, nms_iou_threshold)
    return keep