# Adapted from torchvision/models/detection/_utils.py 
from abc import ABCMeta, abstractmethod
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, FrozenBatchNorm2d, generalized_box_iou_loss

from ops.boxes import norm_angle

class BalancedPositiveNegativeSampler:
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float) -> None:
        """
        Args:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentage of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Args:
            matched_idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)
            neg_idx_per_image_mask = torch.zeros_like(matched_idxs_per_image, dtype=torch.uint8)

            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx


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
    
    @abstractmethod
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
    
    @abstractmethod
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
    
class Matcher:
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        "BELOW_LOW_THRESHOLD": int,
        "BETWEEN_THRESHOLDS": int,
    }

    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False) -> None:
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        torch._assert(low_threshold <= high_threshold, "low_threshold should be <= high_threshold")
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images during training")
            else:
                raise ValueError("No proposal boxes available for one of the images during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None  # type: ignore[assignment]

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (matched_vals < self.high_threshold)
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            if all_matches is None:
                torch._assert(False, "all_matches should not be None")
            else:
                self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches: Tensor, all_matches: Tensor, match_quality_matrix: Tensor) -> None:
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has the highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find the highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(match_quality_matrix == highest_quality_foreach_gt[:, None])
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

def _box_loss(
    type: str,
    box_coder: BoxCoder,
    anchors_per_image: Tensor,
    matched_gt_boxes_per_image: Tensor,
    bbox_regression_per_image: Tensor,
    cnf: Optional[Dict[str, float]] = None,
) -> Tensor:
    torch._assert(type in ["l1", "smooth_l1", "ciou", "diou", "giou"], f"Unsupported loss: {type}")

    if type == "l1":
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        return F.l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
    elif type == "smooth_l1":
        target_regression = box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        beta = cnf["beta"] if cnf is not None and "beta" in cnf else 1.0
        return F.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum", beta=beta)
    else:
        bbox_per_image = box_coder.decode_single(bbox_regression_per_image, anchors_per_image)
        eps = cnf["eps"] if cnf is not None and "eps" in cnf else 1e-7
        if type == "ciou":
            return complete_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)
        if type == "diou":
            return distance_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)
        # otherwise giou
        return generalized_box_iou_loss(bbox_per_image, matched_gt_boxes_per_image, reduction="sum", eps=eps)
