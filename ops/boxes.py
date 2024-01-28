import torch
from torch import Tensor

from ops._box_convert import (
    poly2hbb_np,
    poly2obb_np,
    obb2poly_np,
    obb2poly,
)
from detectron2._C import (
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

def box_iou_rotated(boxes1: Tensor, boxes2: Tensor, angle_aware: bool = False) -> Tensor:
    """ Rotated box IoU. mode_flag and aligned are kept for compatibility with mmrotate implementation.
    Args:
        boxes1, boxes2 (Tensor[N, 5]): boxes in ``(cx, cy, w, h, a)`` format
        angle_aware: arIoU introduced in https://arxiv.org/pdf/1711.09405.pdf
                     if angle_aware=True, IoU *= max(0, cos(box1_a - box2_a)), 
                     i.e. modulates IoU by a factor <= 1 and suppresses IoU when angle difference is within (90, 270)
    Returns:
        Tensor[N, N]: the NxN matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    return _C_box_iou_rotated(boxes1, boxes2, angle_aware)

# Adapted from https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/nms.py
def nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, angle_aware: bool):
    """
    Performs non-maximum suppression (NMS) on the rotated boxes according
    to their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Note that RotatedBox (5, 3, 4, 2, -90) covers exactly the same region as
    RotatedBox (5, 3, 4, 2, 90) does, and their IoU will be 1. However, they
    can be representing completely different objects in certain tasks, e.g., OCR.

    As for the question of whether rotated-NMS should treat them as faraway boxes
    even though their IOU is 1, it depends on the application and/or ground truth annotation.

    As an extreme example, consider a single character v and the square box around it.

    If the angle is 0 degree, the object (text) would be read as 'v';

    If the angle is 90 degrees, the object (text) would become '>';

    If the angle is 180 degrees, the object (text) would become '^';

    If the angle is 270/-90 degrees, the object (text) would become '<'

    All of these cases have IoU of 1 to each other, and rotated NMS that only
    uses IoU as criterion would only keep one of them with the highest score -
    which, practically, still makes sense in most cases because typically
    only one of theses orientations is the correct one. Also, it does not matter
    as much if the box is only used to classify the object (instead of transcribing
    them with a sequential OCR recognition model) later.

    On the other hand, when we use IoU to filter proposals that are close to the
    ground truth during training, we should definitely take the angle into account if
    we know the ground truth is labeled with the strictly correct orientation (as in,
    upside-down words are annotated with -180 degrees even though they can be covered
    with a 0/90/-90 degree box, etc.)

    The way the original dataset is annotated also matters. For example, if the dataset
    is a 4-point polygon dataset that does not enforce ordering of vertices/orientation,
    we can estimate a minimum rotated bounding box to this polygon, but there's no way
    we can tell the correct angle with 100% confidence (as shown above, there could be 4 different
    rotated boxes, with angles differed by 90 degrees to each other, covering the exactly
    same region). In that case we have to just use IoU to determine the box
    proximity (as many detection benchmarks (even for text) do) unless there're other
    assumptions we can make (like width is always larger than height, or the object is not
    rotated by more than 90 degrees CCW/CW, etc.)

    In summary, not considering angles in rotated NMS seems to be a good option for now,
    but we should be aware of its implications.

    Added:
        Above is the original comment from Detectron2. We add an option to make the IoU 
        angle-aware, i.e. IoU *= max(0, cos(box1_a - box2_a)), such that we consider angle 
        difference when computing IoU. This allows to adapt to the above mentioned cases.

    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold
        angle_aware (bool): If True, IoU *= max(0, cos(box1_a - box2_a)), i.e. modulates IoU by a factor <= 1 and suppresses IoU when angle difference is within (90, 270)

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """
    return _C_nms_rotated(boxes, scores, iou_threshold, angle_aware)

@torch.jit.script_if_tracing
def batched_nms_rotated(
    boxes: torch.Tensor, scores: torch.Tensor, idxs: torch.Tensor, iou_threshold: float, angle_aware: bool = False
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
        angle_aware (bool): 
            If True, IoU *= max(0, cos(box1_a - box2_a)), i.e. modulates IoU by a factor <= 1 and suppresses IoU when angle difference is within (90, 270)

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
    keep = nms_rotated(boxes_for_nms, scores, iou_threshold, angle_aware)
    return keep