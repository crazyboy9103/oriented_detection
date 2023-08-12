from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import nn, Tensor
from torchvision.ops import roi_align
from torchvision.models.detection import _utils as det_utils

from ops import boxes as box_ops
# from models.detection import det_utils
from models.detection.box_coders import BoxCoder, HBoxCoder, OBoxCoder


def rotated_fastrcnn_loss(class_logits, hbox_regression, obox_regression, 
                          labels, hbox_regression_targets, obox_regression_targets):
    # type: (Tensor, Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    """
    Computes the loss for Rotated Faster R-CNN.

    Args:
        class_logits (Tensor)
        hbox_regression (Tensor)
        obox_regression (Tensor)
        labels (list[BoxList])
        hbox_regression_targets (Tensor)
        obox_regression_targets (Tensor)
        
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
        obox_loss (Tensor)
    """
   
    N, num_classes = class_logits.shape
    
    box_labels = torch.cat(labels, dim=0)

    def compute_box_loss(regression, labels, regression_targets, horizontal=True):
        regression_targets = torch.cat(regression_targets, dim=0)
        
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]

        box_dim = 4 if horizontal else 5
        regression = regression.reshape(N, regression.size(-1) // box_dim, box_dim)

        box_loss = F.smooth_l1_loss(
            regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum",
        )
        box_loss = box_loss / labels.numel()
        return box_loss
    # Compute for horizontal branch 
    hbox_loss = compute_box_loss(hbox_regression, box_labels, hbox_regression_targets, horizontal=True)
    # Compute for rotated branch
    obox_loss = compute_box_loss(obox_regression, box_labels, obox_regression_targets, horizontal=False)
    classification_loss = F.cross_entropy(class_logits, box_labels)
    return classification_loss, hbox_loss, obox_loss

class RoIHeads(nn.Module):
    __annotations__ = {
        "obox_coder": HBoxCoder,
        "hbox_coder": BoxCoder,
        "proposal_matcher": det_utils.Matcher,
        "fg_bg_sampler": det_utils.BalancedPositiveNegativeSampler,
    }

    def __init__(
        self,
        box_roi_pool,
        box_head,
        box_predictor,
        # Rotated Faster R-CNN training
        fg_iou_thresh,
        bg_iou_thresh,
        batch_size_per_image,
        positive_fraction,
        bbox_reg_weights,
        # Faster R-CNN inference
        score_thresh,
        nms_thresh,
        detections_per_img,
    ):
        super().__init__()

        self.rotated_box_similarity = box_ops.box_iou_rotated
        self.horizontal_box_similarity = box_ops.box_iou
        
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10.0, 10.0, 5.0, 5.0, 10.0)
        
        self.obox_coder = HBoxCoder(bbox_reg_weights)
        self.hbox_coder = BoxCoder(bbox_reg_weights[:4])
        
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def assign_targets_to_proposals(self, proposals: List[Tensor], gt_boxes: List[Tensor], gt_oboxes: List[Tensor], gt_labels: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        matched_idxs = []
        labels = []
        
        for proposals_in_image, gt_boxes_in_image, gt_oboxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_oboxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                horizontal_match_quality_matrix = self.horizontal_box_similarity(gt_boxes_in_image, proposals_in_image)
                # rotated gt_boxes are in rotated format (cx, cy, w, h, a)
                # proposals are in horizontal format (x1, y1, x2, y2)
                # TODO support le90 and le135
                rorated_proposals_in_image = box_ops.hbb2obb(proposals_in_image, version="oc")
                rotated_match_quality_matrix = self.rotated_box_similarity(gt_oboxes_in_image, rorated_proposals_in_image)
                
                assert horizontal_match_quality_matrix.size() == rotated_match_quality_matrix.size()
                match_quality_matrix = (horizontal_match_quality_matrix + rotated_match_quality_matrix) / 2
                
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                
            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            
        return matched_idxs, labels

    def subsample(self, labels: List[Tensor]) -> List[Tensor]:
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.where(pos_inds_img | neg_inds_img)[0]
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals: List[Tensor], gt_boxes: List[Tensor]) -> List[Tensor]:
        proposals = [torch.cat((proposal, gt_box)) for proposal, gt_box in zip(proposals, gt_boxes)]
        return proposals

    def check_targets(self, targets: Optional[List[Dict[str, Tensor]]]) -> None:
        if targets is None:
            raise ValueError("targets should not be None")
        if not all(["bboxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a boxes key")
        if not all(["oboxes" in t for t in targets]):
            raise ValueError("Every element of targets should have a oboxes key")
        if not all(["labels" in t for t in targets]):
            raise ValueError("Every element of targets should have a labels key")

    def select_training_samples(
        self,
        proposals: List[Tensor],
        targets : Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)

        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["bboxes"].to(dtype) for t in targets]
        gt_oboxes = [t["oboxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        # TODO : Is this justified ? 
        # answer: https://github.com/facebookresearch/maskrcnn-benchmark/issues/570#issuecomment-473218934
        # append ground-truth bboxes to propos
        # Seems redundant 
        # Use of horizontal proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        
        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_oboxes, gt_labels)
        # sample a fixed proportion of positive-negative proposals
        sampled_idxs = self.subsample(labels)
        
        matched_gt_hboxes = []
        matched_gt_oboxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_idxs = sampled_idxs[img_id]
            
            proposals[img_id] = proposals[img_id][img_sampled_idxs]
            
            labels[img_id] = labels[img_id][img_sampled_idxs]
                        
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_idxs]
            
            gt_hboxes_in_image = gt_boxes[img_id]
            gt_oboxes_in_image = gt_oboxes[img_id]

            if gt_hboxes_in_image.numel() == 0 or gt_oboxes_in_image.numel() == 0:
                gt_hboxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
                gt_oboxes_in_image = torch.zeros((1, 5), dtype=dtype, device=device)
                
            matched_gt_hboxes.append(gt_hboxes_in_image[matched_idxs[img_id]])
            matched_gt_oboxes.append(gt_oboxes_in_image[matched_idxs[img_id]])
        
        
        horizontal_regression_targets = self.hbox_coder.encode(matched_gt_hboxes, proposals)
        rotated_regression_targets = self.obox_coder.encode(matched_gt_oboxes, proposals)
        return proposals, matched_idxs, labels, horizontal_regression_targets, rotated_regression_targets
    
    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        obox_regression: Tensor,
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]] :
        
        # TODO Use rotated clipping, removing small boxes, NMS for rotated boxes
        # horizontal based can result in misaligned horizontal and rotated boxes
        
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
        
        # print("box_regression", box_regression.shape)
        # print("obox_regression", obox_regression.shape)
        # print("proposal", proposals[0].shape)
        pred_boxes = self.hbox_coder.decode(box_regression, proposals)
        pred_oboxes = self.obox_coder.decode(obox_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)


        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_oboxes_list = pred_oboxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_oboxes = []
        all_scores = []
        all_labels = []
        for boxes, oboxes, scores, image_shape in zip(pred_boxes_list, pred_oboxes_list, pred_scores_list, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            oboxes = oboxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            oboxes = oboxes.reshape(-1, 5)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, oboxes, scores, labels = boxes[inds], oboxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, oboxes, scores, labels = boxes[keep], oboxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, oboxes, scores, labels = boxes[keep], oboxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_oboxes.append(oboxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_oboxes, all_scores, all_labels

    def forward(
        self,
        features: Dict[str, Tensor],
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        targets: Optional[List[Dict[str, Tensor]]] = None,
    ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["bboxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['bboxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.training:
            # TODO : Use different matchers for horizontal and rotated boxes
            # rotated proposals are in rotated format (cx, cy, w, h, a), where a = 0, 
            # as we use horizontal proposals from RPN
            proposals, matched_idxs, labels, horizontal_regression_targets, rotated_regression_targets = self.select_training_samples(proposals, targets)
        
        else:
            labels = None
            horizontal_regression_targets = None
            rotated_regression_targets = None
            matched_idxs = None
        
        # Horizontal ROI Pooling
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, hbox_regression, obox_regression = self.box_predictor(box_features)
        
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            
            if horizontal_regression_targets is None or rotated_regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            
            loss_classifier, loss_box_reg, loss_obox_reg = rotated_fastrcnn_loss(
                class_logits, hbox_regression, obox_regression,
                labels, horizontal_regression_targets, rotated_regression_targets
            )
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_obox_reg": loss_obox_reg}
        else:
            boxes, oboxes, scores, labels = self.postprocess_detections(class_logits, hbox_regression, obox_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "bboxes": boxes[i],
                        "oboxes": oboxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )
            with torch.no_grad():
                loss_classifier, loss_box_reg, loss_obox_reg = rotated_fastrcnn_loss(
                    class_logits, hbox_regression, obox_regression,
                    labels, horizontal_regression_targets, rotated_regression_targets
                )
                losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_obox_reg": loss_obox_reg}

        return result, losses