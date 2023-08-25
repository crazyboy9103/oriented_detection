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

from ops._box_convert import (
    poly2hbb_np,
    poly2obb,
    poly2obb_np,
    obb2poly,
    obb2poly_np,
    obb2xyxy,
    hbb2obb,
)
from detectron2._C import (
    nms_rotated as _C_nms_rotated,
    box_iou_rotated,
    roi_align_rotated_forward,
    roi_align_rotated_backward
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

# Note: this function (nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future
def nms_rotated(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float):
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

    Args:
        boxes (Tensor[N, 5]): Rotated boxes to perform NMS on. They are expected to be in
           (x_center, y_center, width, height, angle_degrees) format.
        scores (Tensor[N]): Scores for each one of the rotated boxes
        iou_threshold (float): Discards all overlapping rotated boxes with IoU < iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of the elements that have been kept
        by Rotated NMS, sorted in decreasing order of scores
    """
    return _C_nms_rotated(boxes, scores, iou_threshold)


# Note: this function (batched_nms_rotated) might be moved into
# torchvision/ops/boxes.py in the future


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
# import torch
# from mmcv.ops import nms_rotated


# def multiclass_nms_rotated(multi_bboxes,
#                            multi_scores,
#                            score_thr,
#                            nms,
#                            max_num=-1,
#                            score_factors=None,
#                            return_inds=False):
#     """NMS for multi-class bboxes.

#     Args:
#         multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
#         multi_scores (torch.Tensor): shape (n, #class), where the last column
#             contains scores of the background class, but this will be ignored.
#         score_thr (float): bbox threshold, bboxes with scores lower than it
#             will not be considered.
#         nms (float): Config of NMS.
#         max_num (int, optional): if there are more than max_num bboxes after
#             NMS, only top max_num will be kept. Default to -1.
#         score_factors (Tensor, optional): The factors multiplied to scores
#             before applying NMS. Default to None.
#         return_inds (bool, optional): Whether return the indices of kept
#             bboxes. Default to False.

#     Returns:
#         tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
#         (k), and (k). Dets are boxes with scores. Labels are 0-based.
#     """
#     num_classes = multi_scores.size(1) - 1
#     # exclude background category
#     if multi_bboxes.shape[1] > 5:
#         bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
#     else:
#         bboxes = multi_bboxes[:, None].expand(
#             multi_scores.size(0), num_classes, 5)
#     scores = multi_scores[:, :-1]

#     labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
#     labels = labels.view(1, -1).expand_as(scores)
#     bboxes = bboxes.reshape(-1, 5)
#     scores = scores.reshape(-1)
#     labels = labels.reshape(-1)

#     # remove low scoring boxes
#     valid_mask = scores > score_thr
#     if score_factors is not None:
#         # expand the shape to match original shape of score
#         score_factors = score_factors.view(-1, 1).expand(
#             multi_scores.size(0), num_classes)
#         score_factors = score_factors.reshape(-1)
#         scores = scores * score_factors

#     inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
#     bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

#     if bboxes.numel() == 0:
#         dets = torch.cat([bboxes, scores[:, None]], -1)
#         if return_inds:
#             return dets, labels, inds
#         else:
#             return dets, labels

#     # Strictly, the maximum coordinates of the rotating box (x,y,w,h,a)
#     # should be calculated by polygon coordinates.
#     # But the conversion from rbbox to polygon will slow down the speed.
#     # So we use max(x,y) + max(w,h) as max coordinate
#     # which is larger than polygon max coordinate
#     # max(x1, y1, x2, y2,x3, y3, x4, y4)
#     max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
#     offsets = labels.to(bboxes) * (max_coordinate + 1)
#     if bboxes.size(-1) == 5:
#         bboxes_for_nms = bboxes.clone()
#         bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
#     else:
#         bboxes_for_nms = bboxes + offsets[:, None]
#     _, keep = nms_rotated(bboxes_for_nms, scores, nms.iou_thr)

#     if max_num > 0:
#         keep = keep[:max_num]

#     bboxes = bboxes[keep]
#     scores = scores[keep]
#     labels = labels[keep]

#     if return_inds:
#         return torch.cat([bboxes, scores[:, None]], 1), labels, keep
#     else:
#         return torch.cat([bboxes, scores[:, None]], 1), labels