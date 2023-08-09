from typing import List, Tuple
from abc import ABCMeta, abstractmethod
import math

import torch
from torch import Tensor 

class BaseBoxCoder(metaclass=ABCMeta):
    """
    Abstract base class for box encoders.
    
    Args:
        weights (List[float]): weights for ``(x, y, w, h, (optional) a)`` used during encoding and decoding
    """
    def __init__(self, weights: List[float] = (1.0, 1.0, 1.0, 1.0)):
        self.weights = weights
    
    def _make_weights_compatible(self, gt_bboxes: Tensor):
        dtype = gt_bboxes.dtype
        device = gt_bboxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        return weights
    
    def encode(self, gt_bboxes: List[Tensor], bboxes: List[Tensor]) -> List[Tensor]:
        """
        Encode a set of proposals with respect to some
        reference boxes

        Args:
            gt_bboxes (list[Tensor]): boxes to be encoded w.r.t. bboxes (e.g. object locations)
            bboxes (list[Tensor]): reference boxes (e.g. anchors)
            
        Returns:
            targets (list[Tensor]): encoded boxes (e.g. object locations w.r.t. anchors)
        """
        boxes_per_image = [b.size(0) for b in gt_bboxes]
        gt_bboxes = torch.cat(gt_bboxes, dim=0)
        bboxes = torch.cat(bboxes, dim=0)
        weights = self._make_weights_compatible(gt_bboxes)
        targets = self.encode_single(gt_bboxes, bboxes, weights)
        return targets.split(boxes_per_image, 0)
    
    @abstractmethod
    def encode_single(self, gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
        """
        Args:
            gt_bboxes (Tensor): concatenated gt_bboxes
            bboxes (Tensor): concatenated reference boxes
            weights (Tensor): weights for ``(x, y, w, h, (optional) a)``
            
        Returns:
            targets (Tensor): concatenated encoded boxes 
        """
        raise NotImplementedError
    
    def decode(self, pred_bboxes: Tensor, bboxes: List[Tensor]) -> Tensor:
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Args:
            pred_bboxes (Tensor): encoded boxes (e.g. deltas w.r.t. anchors = targets)
            bboxes (List[Tensor]): reference boxes (e.g. anchors)
            
        Returns:
            decoded_boxes (Tensor): decoded boxes (e.g. actual bbox locations)
        """
        torch._assert(
            isinstance(bboxes, (list, tuple)),
            "This function expects boxes of type list or tuple.",
        )
        torch._assert(
            isinstance(pred_bboxes, torch.Tensor),
            "This function expects pred_bboxes of type torch.Tensor.",
        )
        boxes_per_image = [b.size(0) for b in bboxes]
        concat_boxes = torch.cat(bboxes, dim=0)
        box_sum = 0
        for val in boxes_per_image:
            box_sum += val
        if box_sum > 0:
            pred_bboxes = pred_bboxes.reshape(box_sum, -1)
            
        concat_boxes = concat_boxes.to(pred_bboxes.dtype)
        weights = self._make_weights_compatible(concat_boxes)
        
        pred_bboxes = self.decode_single(pred_bboxes, concat_boxes, weights, box_sum)
        return pred_bboxes

    @abstractmethod
    def decode_single(self, pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, box_sum: int) -> Tensor:
        """
        Args:
            pred_bboxes (Tensor): concatenated encoded boxes 
            bboxes (Tensor): concatenated reference boxes
            weights (Tensor): weights for ``(x, y, w, h, (optional) a)``
            box_sum (int): number of boxes
        Returns:
            decoded_boxes (Tensor): concatenated decoded boxes 
        """
        raise NotImplementedError

@torch.jit._script_if_tracing
def encode_hboxes(gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes
    ! IMPORTANT !
    This function presumes that the reference boxes are horizontal and the gt boxes are rotated.

    Args:
        gt_bboxes (Tensor[-1, 5]): rotated reference boxes ``(cx, cy, w, h, a)``
        bboxes (Tensor[-1, 4]): rotated boxes to be encoded ``(x1, y1, x2, y2)``
        weights (Tensor[5]): the weights for ``(x, y, w, h, a)``
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]
    wa = weights[4]
    
    bboxes_x1 = bboxes[:, 0].unsqueeze(1)
    bboxes_y1 = bboxes[:, 1].unsqueeze(1)
    bboxes_x2 = bboxes[:, 2].unsqueeze(1)
    bboxes_y2 = bboxes[:, 3].unsqueeze(1)

    gt_ctr_x = gt_bboxes[:, 0].unsqueeze(1)
    gt_ctr_y = gt_bboxes[:, 1].unsqueeze(1)
    gt_widths = gt_bboxes[:, 2].unsqueeze(1)
    gt_heights = gt_bboxes[:, 3].unsqueeze(1)
    gt_angles = gt_bboxes[:, 4].unsqueeze(1)
    
    # implementation starts here
    ex_widths = bboxes_x2 - bboxes_x1
    ex_heights = bboxes_y2 - bboxes_y1
    ex_ctr_x = bboxes_x1 + 0.5 * ex_widths
    ex_ctr_y = bboxes_y1 + 0.5 * ex_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)
    targets_da = wa * gt_angles
    
    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh, targets_da), dim=1)
    return targets

@torch.jit._script_if_tracing
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
    
    # Distance from center to box's corner.
    c_to_c_h = torch.tensor(0.5, dtype=pred_ctr_y.dtype, device=pred_h.device) * pred_h
    c_to_c_w = torch.tensor(0.5, dtype=pred_ctr_x.dtype, device=pred_w.device) * pred_w

    pred_boxes1 = pred_ctr_x - c_to_c_w
    pred_boxes2 = pred_ctr_y - c_to_c_h
    pred_boxes3 = pred_ctr_x + c_to_c_w
    pred_boxes4 = pred_ctr_y + c_to_c_h
    pred_a = da 
    pred_boxes = torch.stack((pred_boxes1, pred_boxes2, pred_boxes3, pred_boxes4, pred_a), dim=2).flatten(1)
    return pred_boxes 

class HBoxCoder(BaseBoxCoder):
    def __init__(self, weights: Tuple[float, float, float, float, float], bbox_xform_clip: float = math.log(1000.0 / 16)):
        """
        Encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh, da), and
        decodes delta (dx, dy, dw, dh, da) back to obox (cx, cy, w, h, a).
        
        Example: uses horizontal proposals (x1, y1, x2, y2) to encode
        rotated ground-truth boxes (cx, cy, w, h, a) into delta (dx, dy, dw, dh, da).
        
        Args:
            weights (Tuple[float, float, float, float, float]): weights for (dx, dy, dw, dh, da)
        """
        super(HBoxCoder, self).__init__(weights)
        self.bbox_xform_clip = bbox_xform_clip

    def encode_single(self, gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
        assert gt_bboxes.size(0) == bboxes.size(0)
        assert gt_bboxes.size(-1) == 5
        assert bboxes.size(-1) == 4
        targets = encode_hboxes(gt_bboxes, bboxes, weights)
        return targets
    
    def decode_single(self, pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, box_sum: int) -> Tensor:
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        pred_boxes = decode_hboxes(pred_bboxes, bboxes, weights, self.bbox_xform_clip)
        if box_sum > 0:
            pred_bboxes = pred_bboxes.reshape(box_sum, -1, 5)
        return pred_boxes

class OBoxCoder(BaseBoxCoder):
    def __init__(self, weights: Tuple[float, float, float, float, float]):
        """
        Encodes obox (cx, cy, w, h, a) into delta (dx, dy, dw, dh, da), and
        decodes delta (dx, dy, dw, dh, da) back to obox (cx, cy, w, h, a).
        
        Example: uses rotated proposals (cx, cy, w, h, a) to encode
        rotated ground-truth boxes (cx, cy, w, h, a) into delta (dx, dy, dw, dh, da).
        
        Args:
            weights (Tuple[float, float, float, float, float]): weights for (dx, dy, dw, dh, da)
        """
        super(OBoxCoder, self).__init__(weights)

    def encode_single(self, gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
        assert gt_bboxes.size(0) == bboxes.size(0)
        assert gt_bboxes.size(-1) == 5
        assert bboxes.size(-1) == 5
        # TODO: implement encode_oboxes
        # targets = encode_oboxes(gt_bboxes, bboxes, weights)
        targets = None
        return targets
    
    def decode_single(self, pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, box_sum: int) -> Tensor:
        assert pred_bboxes.size(0) == bboxes.size(0)
        if pred_bboxes.ndim == 3:
            assert pred_bboxes.size(1) == bboxes.size(1)
        # TODO: implement decode_oboxes
        # pred_boxes = decode_oboxes(pred_bboxes, bboxes, weights, self.bbox_xform_clip)
        if box_sum > 0:
            pred_bboxes = pred_bboxes.reshape(box_sum, -1, 5)
        return pred_bboxes
 
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
        self.bbox_xform_clip = bbox_xform_clip

    def encode_single(self, gt_bboxes: Tensor, bboxes: Tensor, weights: Tensor) -> Tensor:
        targets = encode_boxes(gt_bboxes, bboxes, weights)
        return targets

    def decode_single(self, pred_bboxes: Tensor, bboxes: Tensor, weights: Tensor, box_sum: int) -> Tensor:
        pred_boxes = decode_boxes(pred_bboxes, bboxes, weights, self.bbox_xform_clip)
        if box_sum > 0:
            pred_bboxes = pred_bboxes.reshape(box_sum, -1, 4)
        return pred_boxes