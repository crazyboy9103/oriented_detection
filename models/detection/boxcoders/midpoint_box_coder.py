from typing import Tuple

import torch
from torch import Tensor

from .base import BaseBoxCoder
from ops import boxes as box_ops

def encode_midpoint_boxes(gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes

    Args:
        gt_bboxes (Tensor[-1, 5]): rotated reference boxes ``(cx, cy, w, h, a)`` # angle
        bboxes (Tensor[-1, 4]): boxes to be encoded ``(x1, y1, x2, y2)``
        weights (Tensor[6]): the weights for ``(cx, cy, w, h, a, b)`` # alpha, beta 
    
    Returns:
        targets (Tensor[-1, 6]): encoded boxes ``(dx, dy, dw, dh, da, db)``. 
        Note that da, db are alpha and beta, the offsets from midpoints of top and right edges to the vertices of rotated box.
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    wa = weights[4]
    wb = weights[5]
    
    ex_ctr_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    ex_ctr_y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    ex_widths = bboxes[:, 2] - bboxes[:, 0]
    ex_heights = bboxes[:, 3] - bboxes[:, 1]

    hbb = box_ops.obb2xyxy(gt_bboxes)
    poly = box_ops.obb2poly(gt_bboxes)
    
    gt_ctr_x = (hbb[:, 0] + hbb[:, 2]) * 0.5
    gt_ctr_y = (hbb[:, 1] + hbb[:, 3]) * 0.5
    gt_widths = hbb[:, 2] - hbb[:, 0]
    gt_heights = hbb[:, 3] - hbb[:, 1]
    
    x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]
    y_min, _ = torch.min(y_coor, dim=1, keepdim=True)
    x_max, _ = torch.max(x_coor, dim=1, keepdim=True)

    _x_coor = x_coor.clone()
    _x_coor[torch.abs(y_coor - y_min) > 0.1] = -1000
    gt_alpha, _ = torch.max(_x_coor, dim=1)

    _y_coor = y_coor.clone()
    _y_coor[torch.abs(x_coor - x_max) > 0.1] = -1000
    gt_beta, _ = torch.max(_y_coor, dim=1)

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    
    targets_da = wa * (gt_alpha - gt_ctr_x) / gt_widths # distance to mid point of top edge
    targets_db = wb * (gt_beta - gt_ctr_y) / gt_heights # distance to mid point of right edge
    
    targets_dx = targets_dx.unsqueeze(1)
    targets_dy = targets_dy.unsqueeze(1)
    targets_dw = targets_dw.unsqueeze(1)
    targets_dh = targets_dh.unsqueeze(1)
    targets_da = targets_da.unsqueeze(1)
    targets_db = targets_db.unsqueeze(1)
    targets = torch.cat([targets_dx, targets_dy, targets_dw, targets_dh, targets_da, targets_db], dim=1)
    return targets

def decode_midpoint_boxes(pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, bbox_xform_clip: float) -> Tensor:
    ctr_x = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
    ctr_y = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
    widths = bboxes[:, 2] - bboxes[:, 0]
    heights = bboxes[:, 3] - bboxes[:, 1]
    ctr_x = ctr_x[:, None]
    ctr_y = ctr_y[:, None]
    widths = widths[:, None]
    heights = heights[:, None]
    
    wx, wy, ww, wh, wa, wb = weights
    dx = pred_bboxes[:, 0::6] / wx
    dy = pred_bboxes[:, 1::6] / wy
    dw = pred_bboxes[:, 2::6] / ww
    dh = pred_bboxes[:, 3::6] / wh
    da = pred_bboxes[:, 4::6] / wa
    db = pred_bboxes[:, 5::6] / wb
    
    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=bbox_xform_clip)
    dh = torch.clamp(dh, max=bbox_xform_clip)
   
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    
    x1 = pred_ctr_x - pred_w * 0.5
    y1 = pred_ctr_y - pred_h * 0.5
    x2 = pred_ctr_x + pred_w * 0.5
    y2 = pred_ctr_y + pred_h * 0.5
    
    da = da.clamp(min=-0.5, max=0.5)
    db = db.clamp(min=-0.5, max=0.5)

    pred_a = pred_ctr_x + da * pred_w
    _pred_a = pred_ctr_x - da * pred_w
    pred_b = pred_ctr_y + db * pred_h
    _pred_b = pred_ctr_y - db * pred_h
    
    polys = torch.stack([pred_a, y1, x2, pred_b, _pred_a, y2, x1, _pred_b], dim=-1) 
    center = torch.stack([pred_ctr_x, pred_ctr_y, pred_ctr_x, pred_ctr_y, pred_ctr_x, pred_ctr_y, pred_ctr_x, pred_ctr_y], dim=-1)
    center_polys = polys - center
    
    # shorter diagonal to match longer diagonal
    diag_len = torch.sqrt(center_polys[..., 0::2] ** 2 + center_polys[..., 1::2] ** 2)
    max_diag_len, _ = torch.max(diag_len, dim=-1, keepdim=True)
    diag_scale_factor = max_diag_len / diag_len
    center_polys = center_polys * diag_scale_factor.repeat_interleave(2, dim=-1)
    rectpolys = center_polys + center
    
    pred_boxes = box_ops.poly2obb(rectpolys)
    return pred_boxes

class XYWHAB_XYWHA_BoxCoder(BaseBoxCoder):
    def __init__(self, weights: Tuple[float, float, float, float, float, float]):
        """
        Encodes rotated box (cx, cy, w, h, a) into delta (dx, dy, dw, dh, da, db) using (x1, y1, x2, y2) anchor,
        decodes delta (dx, dy, dw, dh, da, db) to obox (cx, cy, w, h, a).
        # da, db are alpha and beta, the offsets from midpoints of top and right edges to the vertices of rotated box.
        # a is the obb angle. 
        
        Example: uses horizontal proposals (x1, y1, x2, y2) to encode
        rotated ground-truth boxes (cx, cy, w, h, a) into delta (dx, dy, dw, dh, da, db).
        
        Args:
            weights (Tuple[float, float, float, float, float, float]): weights for (dx, dy, dw, dh, da, db)
        """
        super(XYWHAB_XYWHA_BoxCoder, self).__init__(weights)
        
    def encode_single(self, gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
        assert gt_bboxes.size(0) == bboxes.size(0)
        assert gt_bboxes.size(-1) == 5
        assert bboxes.size(-1) == 4
        targets = encode_midpoint_boxes(gt_bboxes, bboxes, weights)
        return targets
    
    def decode_single(self, pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, box_sum: int) -> Tensor:
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert pred_bboxes.size(-1) % 6 == 0
        assert bboxes.size(-1) == 4
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        pred_boxes = decode_midpoint_boxes(pred_bboxes, bboxes, weights, self.bbox_xform_clip)
        if box_sum > 0:
            pred_boxes = pred_boxes.reshape(box_sum, -1, 5)
        return pred_boxes
