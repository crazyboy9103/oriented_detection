from typing import List
from abc import ABCMeta, abstractmethod
import math

import torch
from torch import Tensor

class BaseBoxCoder(metaclass=ABCMeta):
    """
    Abstract base class for boxcoders.
    
    Args:
        weights (List[float]): weights for box parameters used during encoding and decoding
    """
    def __init__(self, weights: List[float] = (1.0, 1.0, 1.0, 1.0), bbox_xform_clip: float = math.log(1000.0 / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip
        
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
        box_sum = sum(boxes_per_image)
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
            weights (Tensor): weights for box parameters used during encoding and decoding
            box_sum (int): number of boxes
        Returns:
            decoded_boxes (Tensor): concatenated decoded boxes 
        """
        raise NotImplementedError
