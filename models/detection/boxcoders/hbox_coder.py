import torch
from torch import Tensor

from .base import BaseBoxCoder

def encode_hboxes(gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes
    
    ! IMPORTANT !
    This function presumes that the reference boxes are horizontal and the gt boxes are rotated.

    Args:
        gt_bboxes (Tensor[-1, 5]): rotated reference boxes (cx, cy, w, h, a)
        bboxes (Tensor[-1, 4]): boxes to be encoded (x1, y1, x2, y2)
        weights (Tensor[5]): the weights for (dx, dy, dw, dh, da)
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    wa = weights[4]
    
    ex_ctr_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    ex_ctr_y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    ex_widths = bboxes[:, 2] - bboxes[:, 0]
    ex_heights = bboxes[:, 3] - bboxes[:, 1]

    gt_ctr_x =   gt_bboxes[:, 0]
    gt_ctr_y =   gt_bboxes[:, 1]
    gt_widths =  gt_bboxes[:, 2]
    gt_heights = gt_bboxes[:, 3]
    gt_angles =  gt_bboxes[:, 4]
    gt_angles = gt_angles % (2 * torch.pi) # angle is already in [0, 2pi), but just in case
    # gt_angles = (gt_angles + torch.pi) % (2 * torch.pi) - torch.pi
    
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    targets_da = wa * (gt_angles)
    
    targets_dx = targets_dx.unsqueeze(1)
    targets_dy = targets_dy.unsqueeze(1)
    targets_dw = targets_dw.unsqueeze(1)
    targets_dh = targets_dh.unsqueeze(1)
    targets_da = targets_da.unsqueeze(1)
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh, targets_da), dim=1)
    return targets

def decode_hboxes(pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, bbox_xform_clip: float) -> Tensor:
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    ctr_x = bboxes[:, 0] + 0.5 * widths
    ctr_y = bboxes[:, 1] + 0.5 * heights
    
    wx, wy, ww, wh, wa = weights
    dx = pred_bboxes[:, 0::5] / wx
    dy = pred_bboxes[:, 1::5] / wy
    dw = pred_bboxes[:, 2::5] / ww
    dh = pred_bboxes[:, 3::5] / wh
    da = pred_bboxes[:, 4::5] / wa
    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)
    
    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]
    # pred_a = (da + torch.pi) % (2 * torch.pi) - torch.pi 
    pred_a = da % (2 * torch.pi)
    
    pred_boxes = torch.stack([pred_ctr_x, pred_ctr_y, pred_w, pred_h, pred_a], dim=2).flatten(1)
    return pred_boxes

class XYXY_XYWHA_BoxCoder(BaseBoxCoder):
    def encode_single(self, gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
        assert gt_bboxes.size(0) == bboxes.size(0)
        assert gt_bboxes.size(-1) == 5
        assert bboxes.size(-1) == 4
        targets = encode_hboxes(gt_bboxes, bboxes, weights)
        return targets
    
    def decode_single(self, pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, box_sum: int) -> Tensor:
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert pred_bboxes.size(-1) % 5 == 0
        assert bboxes.size(-1) == 4
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        pred_boxes = decode_hboxes(pred_bboxes, bboxes, weights, self.bbox_xform_clip)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 5)
        return pred_boxes
