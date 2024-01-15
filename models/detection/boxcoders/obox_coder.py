from typing import Tuple

import torch
from torch import Tensor

from .base import BaseBoxCoder

def encode_oboxes(gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor, proj_xy: bool = True) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes
    
    ! IMPORTANT !
    Rest box coders use (x1, y1, x2, y2) format for bboxes, but this one uses (cx, cy, w, h, a).
    This is due to the fact that Oriented RPN outputs rotated proposals, unlike the standard RPN used in Faster R-CNN.

    Args:
        gt_bboxes (Tensor[-1, 5]): rotated reference boxes ``(cx, cy, w, h, a)``
        bboxes (Tensor[-1, 5]): boxes to be encoded !!``(cx, cy, w, h, a)``!!
        weights (Tensor[5]): the weights for ``(x, y, w, h, a)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    wa = weights[4]
    
    ex_ctr_x = bboxes[:, 0]
    ex_ctr_y = bboxes[:, 1]
    ex_widths = bboxes[:, 2]
    ex_heights = bboxes[:, 3]
    ex_angles = bboxes[:, 4]

    gt_ctr_x =   gt_bboxes[:, 0]
    gt_ctr_y =   gt_bboxes[:, 1]
    gt_widths =  gt_bboxes[:, 2]
    gt_heights = gt_bboxes[:, 3]
    gt_angles =  gt_bboxes[:, 4]
    
    if proj_xy:
        targets_dx = wx * (torch.cos(ex_angles) * (gt_ctr_x - ex_ctr_x) + torch.sin(ex_angles) * (gt_ctr_y - ex_ctr_y)) / ex_widths
        targets_dy = wy * (-torch.sin(ex_angles) * (gt_ctr_x - ex_ctr_x) + torch.cos(ex_angles) * (gt_ctr_y - ex_ctr_y)) / ex_heights
    else:
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    da = (gt_angles - ex_angles)
    da = (da + torch.pi) % (2 * torch.pi) - torch.pi
    targets_da = wa * da
    
    targets_dx = targets_dx.unsqueeze(1)
    targets_dy = targets_dy.unsqueeze(1)
    targets_dw = targets_dw.unsqueeze(1)
    targets_dh = targets_dh.unsqueeze(1)
    targets_da = targets_da.unsqueeze(1)
    targets = torch.cat([targets_dx, targets_dy, targets_dw, targets_dh, targets_da], dim=1)
    return targets

def decode_oboxes(pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, bbox_xform_clip: float, proj_xy: bool = True) -> Tensor:
    ctr_x = bboxes[:, 0]
    ctr_y = bboxes[:, 1]
    widths = bboxes[:, 2]
    heights = bboxes[:, 3]
    angles = bboxes[:, 4]
    
    wx, wy, ww, wh, wa = weights
    dx = pred_bboxes[:, 0::5] / wx
    dy = pred_bboxes[:, 1::5] / wy
    dw = pred_bboxes[:, 2::5] / ww
    dh = pred_bboxes[:, 3::5] / wh
    da = pred_bboxes[:, 4::5] / wa
    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)
    
    if proj_xy:
        pred_ctr_x = dx * widths[:, None] * torch.cos(angles[:, None]) - dy * heights[:, None] * torch.sin(angles[:, None]) + ctr_x[:, None]
        pred_ctr_y = dx * widths[:, None] * torch.sin(angles[:, None]) + dy * heights[:, None] * torch.cos(angles[:, None]) + ctr_y[:, None] 
    else:
        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]
    pred_a = angles[:, None] + da
    pred_a = (pred_a + torch.pi) % (2 * torch.pi) - torch.pi
    
    pred_boxes = torch.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h, pred_a], dim=2).flatten(1)
    return pred_boxes


class XYWHA_XYWHA_BoxCoder(BaseBoxCoder):
    def __init__(self, weights: Tuple[float, float, float, float, float]):
        """
        Encodes obox (cx, cy, w, h, a) into delta (dx, dy, dw, dh, da), and
        decodes delta (dx, dy, dw, dh, da) back to obox (cx, cy, w, h, a).
        
        Example: uses rotated proposals (cx, cy, w, h, a) to encode
        rotated ground-truth boxes (cx, cy, w, h, a) into delta (dx, dy, dw, dh, da).
        
        Args:
            weights (Tuple[float, float, float, float, float]): weights for (dx, dy, dw, dh, da)
        """
        super(XYWHA_XYWHA_BoxCoder, self).__init__(weights)

    def encode_single(self, gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
        assert gt_bboxes.size(0) == bboxes.size(0)
        assert gt_bboxes.size(-1) == 5
        assert bboxes.size(-1) == 5
        targets = encode_oboxes(gt_bboxes, bboxes, weights, proj_xy=True)
        return targets
    
    def decode_single(self, pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, box_sum: int) -> Tensor:
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        pred_bboxes = decode_oboxes(pred_bboxes, bboxes, weights, self.bbox_xform_clip, proj_xy=True)
        if box_sum > 0:
            pred_bboxes = pred_bboxes.reshape(box_sum, -1, 5)
        return pred_bboxes
 