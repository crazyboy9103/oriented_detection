from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.models.detection import _utils as det_utils

from ops import boxes as box_ops
from models.detection.boxcoders import XYXY_XYWH_BoxCoder, XYXY_XYWHA_BoxCoder, XYWHA_XYWHA_BoxCoder
from models.detection.losses import rotated_fastrcnn_loss

class RoIHeads(nn.Module):
    __annotations__ = {
        "hbox_coder": XYXY_XYWHA_BoxCoder,
        "box_coder": XYXY_XYWH_BoxCoder,
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
            bbox_reg_weights = (10, 10, 5, 5, 10)
        
        self.hbox_coder = XYXY_XYWHA_BoxCoder(bbox_reg_weights)
        self.box_coder = XYXY_XYWH_BoxCoder(bbox_reg_weights[:4])
        
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img
        
        self.nms_thresh_rotated = nms_thresh_rotated

    def assign_targets_to_proposals(self, proposals: List[Tensor], gt_boxes: List[Tensor], gt_labels: List[Tensor], horizontal=True) -> Tuple[List[Tensor], List[Tensor]]:
        matched_idxs = []
        labels = []
        
        box_similarity = self.horizontal_box_similarity if horizontal else self.rotated_box_similarity
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image in zip(proposals, gt_boxes, gt_labels):
            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                if not horizontal:
                    # Need to convert horizontal proposals to rotated proposals with angles=0
                    proposals_in_image = box_ops.hbb2obb(proposals_in_image)
                    
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
        # Avoid in-place operation
        proposals = [proposal.clone() for proposal in proposals]
        
        self.check_targets(targets)

        dtype = proposals[0].dtype

        gt_boxes = [t["bboxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]

        if self.training:
            proposals = self.add_gt_proposals(proposals, gt_boxes)
            
        # get matching gt indices for each proposal
        # horizontal: using horizontal or rotated boxes for matching
        matched_idxs, labels = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, horizontal=True)
        
        # sample a fixed proportion of positive-negative proposals
        sampled_idxs = self.subsample(labels)
        return proposals, labels, matched_idxs, sampled_idxs
    
    def filter_training_samples(
            self,
            proposals,
            labels,
            matched_idxs,
            sampled_idxs,
            targets
        ):
        dtype = proposals[0].dtype
        device = proposals[0].device
         # Is this justified ? => see https://github.com/facebookresearch/maskrcnn-benchmark/issues/570#issuecomment-473218934
        # append ground-truth bboxes to proposals for classifier 
        # (box regressor does not obtain any gradients from them)
        matched_gt_hboxes = []
        matched_gt_oboxes = []
        gt_boxes = [t["bboxes"].to(dtype) for t in targets]
        gt_oboxes = [t["oboxes"].to(dtype) for t in targets]
        
        for img_id, img_sampled_idxs in enumerate(sampled_idxs):
            proposals[img_id] = proposals[img_id][img_sampled_idxs]
            labels[img_id] = labels[img_id][img_sampled_idxs]
            
            img_gt_hboxes = gt_boxes[img_id]
            if img_gt_hboxes.numel() == 0:
                img_gt_hboxes = torch.zeros((1, 4), dtype=dtype, device=device)
                
            img_gt_oboxes = gt_oboxes[img_id]
            if img_gt_oboxes.numel() == 0:
                img_gt_oboxes = torch.zeros((1, 5), dtype=dtype, device=device)
                
            matched_gt_hboxes.append(img_gt_hboxes[matched_idxs[img_id][img_sampled_idxs]])
            matched_gt_oboxes.append(img_gt_oboxes[matched_idxs[img_id][img_sampled_idxs]])
        
        horizontal_regression_targets = self.box_coder.encode(matched_gt_hboxes, proposals)
        rotated_regression_targets = self.hbox_coder.encode(matched_gt_oboxes, proposals)
        return proposals, labels, horizontal_regression_targets, rotated_regression_targets
    
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
        box_clip = box_ops.clip_boxes_to_image if horizontal else lambda x, y: x # TODO: clipping for rotated boxes
        box_coder = self.box_coder if horizontal else self.hbox_coder
        nms = box_ops.batched_nms if horizontal else box_ops.batched_nms_rotated
        nms_thresh = self.nms_thresh if horizontal else self.nms_thresh_rotated
        remove_small_fn = box_ops.remove_small_boxes if horizontal else box_ops.remove_small_rotated_boxes
        
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
            boxes = box_clip(boxes, image_shape)

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
            keep = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[keep, :], scores[keep], labels[keep]
            # remove empty boxes
            keep = remove_small_fn(boxes, min_size=1e-2)
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

        train_proposals, train_labels, matched_idxs, sampled_idxs = self.select_training_samples(proposals, targets)
        train_proposals, train_labels, horizontal_regression_targets, rotated_regression_targets = self.filter_training_samples(train_proposals, train_labels, matched_idxs, sampled_idxs, targets)
        
        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if train_labels is None:
                raise ValueError("labels cannot be None")
            
            if horizontal_regression_targets is None or rotated_regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            
            # Horizontal ROI Pooling
            box_features = self.box_roi_pool(features, train_proposals, image_shapes)
            # Shared head 
            box_features = self.box_head(box_features)
            class_logits, hbox_regression, obox_regression = self.box_predictor(box_features)
            
            loss_classifier, loss_box_reg, loss_obox_reg = rotated_fastrcnn_loss(
                class_logits, hbox_regression, obox_regression,
                train_labels, horizontal_regression_targets, rotated_regression_targets
            )
        else:
            # Horizontal ROI Pooling
            box_features = self.box_roi_pool(features, proposals, image_shapes)
            # Shared head 
            box_features = self.box_head(box_features)
            class_logits, hbox_regression, obox_regression = self.box_predictor(box_features)
            
            hboxes, hscores, hlabels = self.postprocess_detections(class_logits, hbox_regression, proposals, image_shapes, horizontal=True)
            oboxes, oscores, olabels = self.postprocess_detections(class_logits, obox_regression, proposals, image_shapes, horizontal=False)
            for i in range(len(hboxes)):
                result.append(
                    {
                        "bboxes": hboxes[i],
                        "labels": hlabels[i],
                        "scores": hscores[i],
                        
                        "oboxes": oboxes[i],
                        "olabels": olabels[i],
                        "oscores": oscores[i],
                    }
                )
    
            valid_logits = []
            valid_hbox = []
            valid_obox = []
            for img_id, img_sampled_idxs in enumerate(sampled_idxs):
                mapped_idxs = (img_id + 1) * img_sampled_idxs
                valid_logits.append(class_logits[mapped_idxs])
                valid_hbox.append(hbox_regression[mapped_idxs])
                valid_obox.append(obox_regression[mapped_idxs])

            valid_logits = torch.cat(valid_logits)
            valid_hbox = torch.cat(valid_hbox)
            valid_obox = torch.cat(valid_obox)
            
            loss_classifier, loss_box_reg, loss_obox_reg = rotated_fastrcnn_loss(
                valid_logits, valid_hbox, valid_obox,
                train_labels, horizontal_regression_targets, rotated_regression_targets
            )

        
        losses = {
            "loss_classifier": loss_classifier, 
            "loss_box_reg": loss_box_reg, 
            "loss_obox_reg": loss_obox_reg
        }

        return result, losses