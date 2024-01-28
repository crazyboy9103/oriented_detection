from typing import Tuple

import torch
from torch import Tensor

from .base import BaseBoxCoder

def encode_oboxes(gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor, epsilon: float = 1e-6) -> Tensor:
    """
    Encode a set of rotated gt boxes with respect to rotated anchors. 
    The transformation is parametrized by 5 deltas (dx, dy, dw, dh, dsina, dcosa) to be learned by a regressor.
    
    ! IMPORTANT !
    This coder uses (cx, cy, w, h, a).

    Args:
        gt_bboxes (Tensor[-1, 5]): rotated reference boxes ``(cx, cy, w, h, a)``
        bboxes (Tensor[-1, 5]): boxes to be encoded ``(cx, cy, w, h, a)``
        weights (Tensor[5]): the weights for ``(x, y, w, h, a)``
        epsilon (float): a small value to avoid division by zero
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    wa = weights[4]
    
    ex_ctr_x, ex_ctr_y, ex_widths, ex_heights, ex_angles = bboxes.unbind(1)
    sin_ex_angles, cos_ex_angles = torch.sin(torch.deg2rad(ex_angles)), torch.cos(torch.deg2rad(ex_angles))
        
    gt_ctr_x, gt_ctr_y, gt_widths, gt_heights, gt_angles = gt_bboxes.unbind(1)
    
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    
    # We normalize sin and cos to be in [-1, +1] to impose sin^2 + cos^2 = 1 constraint
    length = torch.sqrt(sin_ex_angles ** 2 + cos_ex_angles ** 2 + epsilon)
    sin_ex_angles = sin_ex_angles / length
    cos_ex_angles = cos_ex_angles / length

    # With gt_angles, we compute sin and cos 
    sin_gt_angles = torch.sin(torch.deg2rad(gt_angles))
    cos_gt_angles = torch.cos(torch.deg2rad(gt_angles))

    # Then we compute the difference between the two
    targets_sin_da = wa * (sin_gt_angles - sin_ex_angles)
    targets_cos_da = wa * (cos_gt_angles - cos_ex_angles)

    targets_dx = targets_dx.unsqueeze(1)
    targets_dy = targets_dy.unsqueeze(1)
    targets_dw = targets_dw.unsqueeze(1)
    targets_dh = targets_dh.unsqueeze(1)
    targets_sin_da = targets_sin_da.unsqueeze(1)
    targets_cos_da = targets_cos_da.unsqueeze(1)

    targets = torch.cat([targets_dx, targets_dy, targets_dw, targets_dh, targets_sin_da, targets_cos_da], dim=1)
    return targets

def decode_oboxes(pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, bbox_xform_clip: float, epsilon: float = 1e-6) -> Tensor:
    ctr_x, ctr_y, widths, heights, angles = bboxes.unbind(1)
    sin_angles, cos_angles = torch.sin(torch.deg2rad(angles)), torch.cos(torch.deg2rad(angles))
    
    wx, wy, ww, wh, wa = weights
    dx = pred_bboxes[:, 0::6] / wx
    dy = pred_bboxes[:, 1::6] / wy
    dw = pred_bboxes[:, 2::6] / ww
    dh = pred_bboxes[:, 3::6] / wh
    sin_da = pred_bboxes[:, 4::6] / wa
    cos_da = pred_bboxes[:, 5::6] / wa

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)
    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]
    
    pred_sin_a = sin_da + sin_angles[:, None]
    pred_cos_a = cos_da + cos_angles[:, None]

    length = torch.sqrt(pred_sin_a ** 2 + pred_cos_a ** 2 + epsilon)
    pred_sin_a = pred_sin_a / length
    pred_cos_a = pred_cos_a / length

    pred_a = torch.rad2deg(torch.atan2(pred_sin_a, pred_cos_a))
    pred_a = torch.where(pred_a < 0, pred_a + 360.0, pred_a) # make it in [0, 360)
    pred_boxes = torch.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h, pred_a], dim=2).flatten(1)
    return pred_boxes


class XYWHA_XYWHA_BoxCoder(BaseBoxCoder):
    def __init__(self, weights: Tuple[float, float, float, float, float]):
        """
        Encodes obox (cx, cy, w, h, a) => (dx, dy, dw, dh, dsina, dcosa)
        Decodes delta (dx, dy, dw, dh, dsina, dcosa) => (cx, cy, w, h, a).
        
        Args:
            weights (Tuple[float, float, float, float, float]): weights for (dx, dy, dw, dh, da)
                                                                da is applied equally for (dsina, dcosa)
        """
        super(XYWHA_XYWHA_BoxCoder, self).__init__(weights)

    def encode_single(self, gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
        assert gt_bboxes.size(0) == bboxes.size(0)
        assert gt_bboxes.size(-1) == 5
        assert bboxes.size(-1) == 5
        targets = encode_oboxes(gt_bboxes, bboxes, weights)
        return targets
    
    def decode_single(self, pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, box_sum: int) -> Tensor:
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        pred_bboxes = decode_oboxes(pred_bboxes, bboxes, weights, self.bbox_xform_clip)
        if box_sum > 0:
            pred_bboxes = pred_bboxes.reshape(box_sum, -1, 5)
        return pred_bboxes
 