# Essentially equivalent to torchvision BoxCoder
from typing import Tuple
import math

import torch
from torch import Tensor

from .base import BaseBoxCoder

@torch.jit._script_if_tracing
def encode_boxes(gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        gt_bboxes (Tensor[-1, 4]): reference boxes
        bboxes (Tensor[-1, 4]): boxes to be encoded
        weights (Tensor[4]): the weights for ``(x, y, w, h)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    
    bboxes_x1 = bboxes[:, 0].unsqueeze(1)
    bboxes_y1 = bboxes[:, 1].unsqueeze(1)
    bboxes_x2 = bboxes[:, 2].unsqueeze(1)
    bboxes_y2 = bboxes[:, 3].unsqueeze(1)

    gt_bboxes_x1 = gt_bboxes[:, 0].unsqueeze(1)
    gt_bboxes_y1 = gt_bboxes[:, 1].unsqueeze(1)
    gt_bboxes_x2 = gt_bboxes[:, 2].unsqueeze(1)
    gt_bboxes_y2 = gt_bboxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = bboxes_x2 - bboxes_x1
    ex_heights = bboxes_y2 - bboxes_y1
    ex_ctr_x = bboxes_x1 + 0.5 * ex_widths
    ex_ctr_y = bboxes_y1 + 0.5 * ex_heights

    gt_widths = gt_bboxes_x2 - gt_bboxes_x1
    gt_heights = gt_bboxes_y2 - gt_bboxes_y1
    gt_ctr_x = gt_bboxes_x1 + 0.5 * gt_widths
    gt_ctr_y = gt_bboxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets

@torch.jit._script_if_tracing
def decode_boxes(pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, bbox_xform_clip: float) -> Tensor:
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    ctr_x = bboxes[:, 0] + 0.5 * widths
    ctr_y = bboxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = pred_bboxes[:, 0::4] / wx
    dy = pred_bboxes[:, 1::4] / wy
    dw = pred_bboxes[:, 2::4] / ww
    dh = pred_bboxes[:, 3::4] / wh

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    # Distance from center to box's corner.
    c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

    pred_boxes1 = pred_ctr_x - c_to_c_w
    pred_boxes2 = pred_ctr_y - c_to_c_h
    pred_boxes3 = pred_ctr_x + c_to_c_w
    pred_boxes4 = pred_ctr_y + c_to_c_h
    pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4), dim=2).flatten(1)
    return pred_boxes 

class BoxCoder(BaseBoxCoder):
    """
    Equivalent to torchvision BoxCoder

    Args:
        weights (Tuple[float, float, float, float]): weights for (dx, dy, dw, dh)
    """

    def __init__(
        self, weights: Tuple[float, float, float, float], bbox_xform_clip: float = math.log(1000.0 / 16)
    ) -> None:
        """
        Args:
            weights (4-element tuple) : Scaling factors used to scale (dx, dy, dw, dh) deltas
            bbox_xform_clip (float)
        """
        super(BoxCoder, self).__init__(weights)

    def encode_single(self, gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
        targets = encode_boxes(gt_bboxes, bboxes, weights)
        return targets

    def decode_single(self, pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, box_sum: int) -> Tensor:
        pred_boxes = decode_boxes(pred_bboxes, bboxes, weights, self.bbox_xform_clip)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 4)
        return pred_boxes