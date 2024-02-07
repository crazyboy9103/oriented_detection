# Adapted from torchvision/models/detection/rpn.py
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import Conv2dNormActivation
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection import _utils as det_utils

from ops import boxes as box_ops
from models.detection.boxcoders import XYWHA_XYWHA_BoxCoder
from models.detection.anchor_utils import RotatedAnchorGenerator

# RPN Heads
class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
        bbox_dim (int, optional): box dimension, default 4 
    """

    def __init__(self, in_channels: int, num_anchors: int, conv_depth=1, bbox_dim=4) -> None:
        super().__init__()
        self.conv = nn.Sequential(*[
            Conv2dNormActivation(in_channels, in_channels, kernel_size=3, norm_layer=None) 
            for _ in range(conv_depth)
        ])
        
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * bbox_dim, kernel_size=1, stride=1)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight, std=0.01)  # type: ignore[arg-type]
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)  # type: ignore[arg-type]

    def forward(self, features: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in features:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg

# Utility Functions for RPN
def permute_and_flatten(layer: Tensor, N: int, C: int, H: int, W: int) -> Tensor:
    layer = layer.view(N, -1, C, H, W)
    layer = layer.permute(0, 3, 4, 1, 2)
    layer = layer.reshape(N, -1, C)
    return layer


def concat_box_prediction_layers(box_cls: List[Tensor], box_regression: List[Tensor], box_dim: int = 4) -> Tuple[Tensor, Tensor]:
    box_cls_flattened = []
    box_regression_flattened = []
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness and the box_regression
    for box_cls_per_level, box_regression_per_level in zip(box_cls, box_regression):
        # Number of features, number of anchors * number of classes(=2 for rpn), height, width
        N, AxC, H, W = box_cls_per_level.shape
        # Number of anchors * regression box dimension (4 for rotated faster rcnn, 6 for oriented rcnn) 
        AxBoxdim = box_regression_per_level.shape[1]
        A = AxBoxdim // box_dim 
        C = AxC // A
        box_cls_per_level = permute_and_flatten(box_cls_per_level, N, C, H, W)
        box_cls_flattened.append(box_cls_per_level)

        box_regression_per_level = permute_and_flatten(box_regression_per_level, N, box_dim, H, W)
        box_regression_flattened.append(box_regression_per_level)
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = torch.cat(box_cls_flattened, dim=1).flatten(0, -2)
    box_regression = torch.cat(box_regression_flattened, dim=1).reshape(-1, box_dim)
    return box_cls, box_regression

class RotatedRegionProposalNetwork(nn.Module):
    def __init__(
        self, 
        anchor_generator: RotatedAnchorGenerator, 
        head: nn.Module,
        # Faster-RCNN Training
        fg_iou_thresh: float,
        bg_iou_thresh: float,
        batch_size_per_image: int,
        positive_fraction: float,
        # Faster-RCNN Inference
        pre_nms_top_n: Dict[str, int],
        post_nms_top_n: Dict[str, int],
        nms_thresh: float,
        score_thresh: float = 0.0,
    ):
        super().__init__()
        self.anchor_generator = anchor_generator
        self.head = head
        self.box_coder = XYWHA_XYWHA_BoxCoder(weights=(1.0, 1.0, 1.0, 1.0, 1.0))
        
        # used during training
        self.box_similarity = box_ops.box_iou_rotated
        
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=True)
        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.batch_size_per_image = batch_size_per_image
        
        # used during testing
        self._pre_nms_top_n = pre_nms_top_n
        self._post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.min_size = 1e-3
        
        # We would regress 4 coordinates and 2 (sin, cos) angles 
        self.regression_dim = 6
        # Then decode 2 angles to 1 angle using atan2
        self.proposal_dim = 5
        
    def pre_nms_top_n(self) -> int:
        if self.training:
            return self._pre_nms_top_n["training"]
        return self._pre_nms_top_n["testing"]

    def post_nms_top_n(self) -> int:
        if self.training:
            return self._post_nms_top_n["training"]
        return self._post_nms_top_n["testing"]

    def assign_targets_to_anchors(
        self, anchors: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor]]:
        labels = []
        matched_gt_boxes = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            gt_boxes = targets_per_image["oboxes"]

            if gt_boxes.numel() == 0:
                # Background image (negative example)
                device = anchors_per_image.device
                matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
                labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
            else:
                match_quality_matrix = self.box_similarity(gt_boxes, anchors_per_image, True)
                matched_idxs = self.proposal_matcher(match_quality_matrix)
                # get the targets corresponding GT for each proposal
                # NB: need to clamp the indices because we can have a single
                # GT in the image, and matched_idxs can be -2, which goes
                # out of bounds
                matched_gt_boxes_per_image = gt_boxes[matched_idxs.clamp(min=0)]

                labels_per_image = matched_idxs >= 0
                labels_per_image = labels_per_image.to(dtype=torch.float32)

                # Background (negative examples)
                bg_indices = matched_idxs == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_per_image[bg_indices] = 0.0

                # discard indices that are between thresholds
                inds_to_discard = matched_idxs == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1.0

            labels.append(labels_per_image)
            matched_gt_boxes.append(matched_gt_boxes_per_image)
        return labels, matched_gt_boxes

    def _get_top_n_idx(self, objectness: Tensor, num_anchors_per_level: List[int]) -> Tensor:
        r = []
        offset = 0
        for ob in objectness.split(num_anchors_per_level, 1):
            num_anchors = ob.shape[1]
            pre_nms_top_n = det_utils._topk_min(ob, self.pre_nms_top_n(), 1)
            _, top_n_idx = ob.topk(pre_nms_top_n, dim=1)
            r.append(top_n_idx + offset)
            offset += num_anchors
        return torch.cat(r, dim=1)

    def filter_proposals(
        self,
        proposals: Tensor,
        objectness: Tensor,
        image_shapes: List[Tuple[int, int]],
        num_anchors_per_level: List[int],
    ) -> Tuple[List[Tensor], List[Tensor]]:

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        levels = [
            torch.full((n,), idx, dtype=torch.int64, device=device) for idx, n in enumerate(num_anchors_per_level)
        ]
        levels = torch.cat(levels, 0)
        levels = levels.reshape(1, -1).expand_as(objectness)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]

        objectness = objectness[batch_idx, top_n_idx]
        levels = levels[batch_idx, top_n_idx]
        proposals = proposals[batch_idx, top_n_idx]

        objectness_prob = torch.sigmoid(objectness)

        final_boxes = []
        final_scores = []
        
        # TODO: clip rotated boxes to image
        # It is not simple to clip rotated boxes to image, so we just skip it for now
        clip_rotated_boxes_to_image = lambda x, _: x 
        
        for boxes, scores, lvl, img_shape in zip(proposals, objectness_prob, levels, image_shapes):
            boxes = clip_rotated_boxes_to_image(boxes, img_shape)
        
            # remove small boxes
            keep = box_ops.remove_small_rotated_boxes(boxes, self.min_size)
            
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # remove low scoring boxes
            # use >= for Backwards compatibility
            keep = torch.where(scores >= self.score_thresh)[0]
            boxes, scores, lvl = boxes[keep], scores[keep], lvl[keep]

            # non-maximum suppression, independently done per level
            keep = box_ops.batched_nms_rotated(boxes, scores, lvl, self.nms_thresh, True)
          
            # keep only topk scoring predictions
            keep = keep[: self.post_nms_top_n()]
            boxes, scores = boxes[keep], scores[keep]

            final_boxes.append(boxes)
            final_scores.append(scores)
        return final_boxes, final_scores

    
    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.
             
        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.head(features)
        anchors = self.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas, self.regression_dim)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, self.proposal_dim)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if targets is not None:
            losses = self._return_loss(anchors, targets, objectness, pred_bbox_deltas)
            
        else:
            if self.training:
                raise ValueError("targets should not be None")
            
        return boxes, losses

    def _return_loss(self, anchors, targets, objectness, pred_bbox_deltas):
        labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
        regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_coord_reg, loss_rpn_box_angle_reg = self.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_coord_reg": loss_rpn_box_coord_reg,
            "loss_rpn_box_angle_reg": loss_rpn_box_angle_reg,
        }
        return losses
    
    def compute_loss(
        self, objectness: Tensor, pred_bbox_deltas: Tensor, labels: List[Tensor], regression_targets: List[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])

        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds])

        regression_targets = torch.cat(regression_targets, dim=0)

        preds = pred_bbox_deltas[sampled_pos_inds]
        targets = regression_targets[sampled_pos_inds]

        preds_coords = preds[:, :4]
        preds_angles = preds[:, 4:]

        targets_coords = targets[:, :4]
        targets_angles = targets[:, 4:]

        coord_loss = F.smooth_l1_loss(
            preds_coords,
            targets_coords,
            beta=1.0 / 9,
            reduction="sum",
        ) / (sampled_inds.numel())

        # angle_loss = 1-F.cosine_similarity(preds_angles, targets_angles, dim=1)
        # angle_loss = angle_loss.sum() / (sampled_inds.numel())
        angle_loss = F.smooth_l1_loss(
            preds_angles,
            targets_angles,
            beta=1.0,
            reduction="sum",
        ) / (sampled_inds.numel())

        return objectness_loss, coord_loss, angle_loss