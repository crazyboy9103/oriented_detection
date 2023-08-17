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
    
    labels = torch.cat(labels, dim=0)
    
    def compute_box_loss(regression, regression_targets, horizontal=True):
        pos_inds = torch.where(labels > 0)[0]
        labels_pos = labels[pos_inds]
        regression_targets = torch.cat(regression_targets, dim=0)
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        box_dim = 4 if horizontal else 5
        regression = regression.reshape(N, regression.size(-1) // box_dim, box_dim)
        box_loss = F.smooth_l1_loss(
            regression[pos_inds, labels_pos],
            regression_targets[pos_inds],
            beta=1.0,
            reduction="sum",
        )
        box_loss = box_loss / pos_inds.numel()
        return box_loss
    # Compute for horizontal branch 
    hbox_loss = compute_box_loss(hbox_regression, hbox_regression_targets, horizontal=True)
    # Compute for rotated branch
    obox_loss = compute_box_loss(obox_regression, obox_regression_targets, horizontal=False)
    try:
        classification_loss = F.cross_entropy(class_logits, labels)
    except:
        classification_loss = torch.tensor(0.0)
        
    return classification_loss, hbox_loss, obox_loss

class RoIHeads(nn.Module):
    __annotations__ = {
        "hbox_coder": HBoxCoder,
        "box_coder": BoxCoder,
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
        # Rotated Faster R-CNN inference
        nms_thresh_rotated,
        
    ):
        super().__init__()

        self.rotated_box_similarity = box_ops.box_iou_rotated
        self.horizontal_box_similarity = box_ops.box_iou
        
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = det_utils.Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)

        self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(batch_size_per_image, positive_fraction)
        self.batch_size_per_image = batch_size_per_image
        
        if bbox_reg_weights is None:
            bbox_reg_weights = (1, 1, 1, 1, 1)
        
        self.hbox_coder = HBoxCoder(bbox_reg_weights)
        self.box_coder = BoxCoder(bbox_reg_weights[:4])
        
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        
        self.nms_thresh_rotated = nms_thresh_rotated

    @torch.no_grad()
    def assign_targets_to_proposals(self, proposals: List[Tensor], gt_boxes: List[Tensor], gt_labels: List[Tensor], horizontal=True) -> Tuple[List[Tensor], List[Tensor]]:
        matched_idxs = []
        labels = []
        
        box_similarity = self.horizontal_box_similarity if horizontal else self.rotated_box_similarity
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            print("gt_labels_in_image", gt_labels_in_image.shape)
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                if not horizontal:
                    # rotated gt_boxes are in rotated format (cx, cy, w, h, a)
                    # proposals are in horizontal format (x1, y1, x2, y2)
                    # TODO support le90 and le135
                    proposals_in_image = box_ops.hbb2obb(proposals_in_image)
                    # x1, y1, x2, y2 = proposals_in_image[:, 0:1], proposals_in_image[:, 1:2], proposals_in_image[:, 2:3], proposals_in_image[:, 3:4]
                    # cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                    # w_regular = torch.where(w > h, w, h)
                    # h_regular = torch.where(w > h, h, w)
                    # a = torch.zeros_like(cx)
                    # theta_regular = torch.where(w > h, a, a + torch.pi / 2)
                    # proposals_in_image = torch.cat((cx, cy, w_regular, h_regular, theta_regular), 1)
                    
                match_quality_matrix = box_similarity(gt_boxes_in_image, proposals_in_image)
                
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

        # Is this justified ? => see https://github.com/facebookresearch/maskrcnn-benchmark/issues/570#issuecomment-473218934
        # append ground-truth bboxes to propos
        # Use of horizontal proposals
        proposals = self.add_gt_proposals(proposals, gt_boxes)
        
        # get matching gt indices for each proposal
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_oboxes, gt_labels, horizontal=False)
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

            if gt_hboxes_in_image.numel() == 0:
                gt_hboxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            if gt_oboxes_in_image.numel() == 0:
                gt_oboxes_in_image = torch.zeros((1, 5), dtype=dtype, device=device)
                
            matched_gt_hboxes.append(gt_hboxes_in_image[matched_idxs[img_id]])
            matched_gt_oboxes.append(gt_oboxes_in_image[matched_idxs[img_id]])
        
        
        horizontal_regression_targets = self.box_coder.encode(matched_gt_hboxes, proposals)
        rotated_regression_targets = self.hbox_coder.encode(matched_gt_oboxes, proposals)
        return proposals, matched_idxs, labels, horizontal_regression_targets, rotated_regression_targets
    
    def postprocess_detections(
        self,
        class_logits: Tensor,
        box_regression: Tensor,
        proposals: List[Tensor],
        image_shapes: List[Tuple[int, int]],
        horizontal=True
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]] :
        
        # TODO Use rotated clipping, removing small boxes, NMS for rotated boxes
        # horizontal based can result in misaligned horizontal and rotated boxes
        box_dim = 4 if horizontal else 5
        box_coder = self.box_coder if horizontal else self.hbox_coder
        nms = box_ops.batched_nms if horizontal else box_ops.batched_nms_rotated
        nms_thresh = self.nms_thresh if horizontal else self.nms_thresh_rotated
        remove_small = box_ops.remove_small_boxes if horizontal else box_ops.remove_small_rotated_boxes
        
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        pred_boxes = box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)

        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
            if horizontal:
                boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            else:
                pass

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            
            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, box_dim)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            # remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[inds, :], scores[inds], labels[inds]
            # remove empty boxes
            keep = remove_small(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep, :], scores[keep], labels[keep]
             
            # non-maximum suppression, independently done per class
            keep = nms(boxes, scores, labels, nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels = boxes[keep, :], scores[keep], labels[keep]
            # all_labels.append(labels)
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            
        return all_boxes, all_scores, all_labels

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
                if not t["oboxes"].dtype in floating_point_types:
                    raise TypeError(f"target oboxes must of float type, instead got {t['oboxes'].dtype}")
                if not t["bboxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['bboxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        proposals, matched_idxs, labels, horizontal_regression_targets, rotated_regression_targets = self.select_training_samples(proposals, targets)
        # print("proposals", proposals[0].shape)
        # print("matched_idxs", matched_idxs[0].shape)
        # print("labels", labels[0].shape)
        # print("horizontal_regression_targets", horizontal_regression_targets[0].shape)
        # print("rotated_regression_targets", rotated_regression_targets[0].shape)
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
        else:
            hboxes, hscores, hlabels = self.postprocess_detections(class_logits, hbox_regression, proposals, image_shapes, horizontal=True)
            oboxes, oscores, olabels = self.postprocess_detections(class_logits, obox_regression, proposals, image_shapes, horizontal=False)
            for i in range(len(hboxes)):
                result.append(
                    {
                        "hboxes": hboxes[i],
                        "hlabels": hlabels[i],
                        "hscores": hscores[i],
                        
                        "oboxes": oboxes[i],
                        "olabels": olabels[i],
                        "oscores": oscores[i],
                    }
                )
        
        loss_classifier, loss_box_reg, loss_obox_reg = rotated_fastrcnn_loss(
            class_logits, hbox_regression, obox_regression,
            labels, horizontal_regression_targets, rotated_regression_targets
        )
        losses = {
            "loss_classifier": loss_classifier, 
            "loss_box_reg": loss_box_reg, 
            "loss_obox_reg": loss_obox_reg
        }

        return result, losses