from typing import Dict, List

import torch
from torch import Tensor

from ops import boxes as box_ops

# TODO add support for horizontal boxes
class DetectionEvaluator:
    def __init__(self, iou_threshold: float | List[float] = 0.5, rotated: bool = False, num_classes: int = 13):
        """Evaluator for axis-aligned or oriented object detection task. Computes precision, recall, f1-score, mAP and mIoU.

        Args:
            iou_threshold (float | List[float]): 
                If List[float] is given, metrics are computed at each threshold and averaged. 
                For instance, COCO metric is computed at [0.5, 0.55... 0.95] and averaged. 
                Defaults to 0.5.
            rotated (bool): 
                Whether the evaluator is for rotated bounding boxes.
                Defaults to False.
            num_classes (int):
                The number of classes in the dataset, excluding background.
                Defaults to 13.
        """
        self.iou_threshold = iou_threshold if isinstance(iou_threshold, list) else [iou_threshold]
        self.rotated = rotated
        self.num_classes = num_classes
        self.iou_calculator = box_ops.box_iou_rotated if rotated else box_ops.box_iou

        self.reset()

    def reset(self):
        # these contain tensors for each class
        # e.g. self.targets[1] contains a list of tensors containing all ground truth boxes for class 1
        # each index in the list corresponds to a different image
        self.targets = {class_idx: [] for class_idx in range(1, self.num_classes+1)}
        self.detections = {class_idx: [] for class_idx in range(1, self.num_classes+1)}
        self.scores = {class_idx: [] for class_idx in range(1, self.num_classes+1)}
    
    def accumulate(
        self, 
        targets: List[Dict[str, Tensor]],
        detections: List[Dict[str, Tensor]],  
    ):  
        """Accumulate targets and detections by class, for computing metrics later.
        Args:
            targets (List[Dict[str, Tensor]]):
                List of targets for each image. Each target is a dictionary containing the following keys:
                    - Either "bboxes"|"oboxes": Ground truth bounding boxes in the format (x1, y1, x2, y2)|(cx, cy, w, h, theta)
                    - "labels": Ground truth labels
            detections (List[Dict[str, Tensor]]):
                List of detections for each image. Each detection is a dictionary containing the following keys:
                    - Either "bboxes"|"oboxes": Predicted bounding boxes in the format (x1, y1, x2, y2)|(cx, cy, w, h, theta)
                    - "labels": Predicted labels
                    - "scores": Predicted scores
        """
        assert len(targets) == len(detections), "Number of targets and detections must be the same"

        for target, detection in zip(targets, detections):
            gt_bboxes = target["oboxes" if self.rotated else "bboxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu()
            
            dt_bboxes = detection["oboxes" if self.rotated else "bboxes"].detach().cpu()
            dt_labels = detection["olabels" if self.rotated else "labels"].detach().cpu()
            dt_scores = detection["oscores" if self.rotated else "scores"].detach().cpu()
            
            for class_idx in range(1, self.num_classes+1):
                gt_mask = gt_labels == class_idx
                dt_mask = dt_labels == class_idx
                
                class_gt_bboxes = gt_bboxes[gt_mask]
                class_dt_bboxes = dt_bboxes[dt_mask]
                class_dt_scores = dt_scores[dt_mask]
                
                self.targets[class_idx].append(class_gt_bboxes)
                self.detections[class_idx].append(class_dt_bboxes)
                self.scores[class_idx].append(class_dt_scores)
    
    def compute_metrics(self):
        detailed_metrics = {}
        
        aggregate_metrics = torch.zeros(6)
        for iou_threshold in self.iou_threshold:
            # threshold_metrics holds values for precision, recall, f1_score, AP, mean_iou, mean_modulated_iou
            
            # For each class, compute metrics and average them to get the metrics for the threshold
            threshold_metrics = torch.zeros(6)
            for class_idx in range(1, self.num_classes+1):
                metrics = self._compute_class_metrics(class_idx, iou_threshold)
                threshold_metrics += torch.tensor(metrics)
            
            threshold_metrics /= self.num_classes

            precision = threshold_metrics[0].item()
            recall = threshold_metrics[1].item()
            f1_score = threshold_metrics[2].item()
            mAP = threshold_metrics[3].item()
            mIoU = threshold_metrics[4].item()
            mModIoU = threshold_metrics[5].item()
            dtheta = mModIoU / mIoU if mIoU != 0 else 0
            
            detailed_metrics[iou_threshold] = {
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score,
                "mAP": mAP,
                "mIoU": mIoU,
                "mModIoU": mModIoU,
                "dtheta": dtheta
            }
            
            aggregate_metrics += threshold_metrics
        
        # For each threshold, compute metrics and average them to get the mean metrics across all thresholds
        aggregate_metrics /= len(self.iou_threshold)
        
        agg_precision = aggregate_metrics[0].item()
        agg_recall = aggregate_metrics[1].item()
        agg_f1_score = aggregate_metrics[2].item()
        agg_mAP = aggregate_metrics[3].item()
        agg_mIoU = aggregate_metrics[4].item()
        agg_mModIoU = aggregate_metrics[5].item()
        agg_dtheta = agg_mModIoU / agg_mIoU if agg_mIoU != 0 else 0
        aggregate_metrics = {
            "Precision": agg_precision,
            "Recall": agg_recall,
            "F1-Score": agg_f1_score,
            "mAP": agg_mAP,
            "mIoU": agg_mIoU,
            "mModIoU": agg_mModIoU,
            "dtheta": agg_dtheta, 
        }
        return aggregate_metrics, detailed_metrics
        
                
                
    def _compute_class_metrics(self, class_idx: int, iou_threshold: float = 0.5):
        class_gt_bboxes = self.targets[class_idx]
        class_dt_bboxes = self.detections[class_idx]
        class_dt_scores = self.scores[class_idx]
        
        TP = []
        FP = []
        dt_ious = []
        gt_angles = []
        dt_angles = []
        
        # This will be used as TP + FN
        total_num_gt_bboxes = 0
        
        for image_idx, (gt_bboxes, dt_bboxes, dt_scores) in enumerate(zip(class_gt_bboxes, class_dt_bboxes, class_dt_scores)):
            num_gt_bboxes = gt_bboxes.shape[0]
            num_dt_bboxes = dt_bboxes.shape[0]
            
            total_num_gt_bboxes += num_gt_bboxes
            
            if num_gt_bboxes == 0 or num_dt_bboxes == 0:
                continue
            
            # Sort detections by score in descending order
            sorted_indices = torch.argsort(dt_scores, descending=True)
            dt_bboxes = dt_bboxes[sorted_indices]
            dt_scores = dt_scores[sorted_indices]
            
            # To keep track of which ground truth boxes have been assigned
            # GT boxes can only be assigned once
            assigned_gt_boxes = set()
            for dt_bbox in dt_bboxes:
                max_iou = 0
                best_gt_idx = None
                for gt_idx, gt_bbox in enumerate(gt_bboxes):                  
                    iou = self.iou_calculator(dt_bbox[None, :], gt_bbox[None, :]).item()
                    if iou > max_iou:
                        max_iou = iou
                        best_gt_idx = gt_idx

                if max_iou >= iou_threshold:
                    if best_gt_idx in assigned_gt_boxes:
                        FP.append(1)
                        TP.append(0)
                        
                    else:
                        assigned_gt_boxes.add(best_gt_idx)
                        
                        FP.append(0)
                        TP.append(1)
                        dt_ious.append(max_iou)
                        
                        # Compute metric for angle 
                        gt_angles.append(gt_bbox[4])
                        dt_angles.append(dt_bbox[4])
                        
                else:
                    FP.append(1)
                    TP.append(0)
        
        # If there are no true positives, return 0 for all metrics
        if not sum(TP):
            return 0, 0, 0, 0, 0, 0
        
        TP_cumsum = torch.cumsum(torch.as_tensor(TP), dim=0)
        FP_cumsum = torch.cumsum(torch.as_tensor(FP), dim=0)
        recalls = TP_cumsum / (total_num_gt_bboxes + 1e-10)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-10)
        
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
        f1_max_idx = torch.argmax(f1_scores)
        recall = recalls[f1_max_idx].item()
        precision = precisions[f1_max_idx].item()
        f1_score = f1_scores[f1_max_idx].item()  
        
        # Add 0 and 1 to the start and end of the arrays to make sure the curve starts at (0, 1) and ends at (1, 0)
        # This is required for computing the area under the curve      
        recalls = torch.cat([torch.tensor([0]), recalls, torch.tensor([1])])
        precisions = torch.cat([torch.tensor([1]), precisions, torch.tensor([0])])
        
        AP = torch.trapz(precisions, recalls).item()
        
        dt_ious = torch.as_tensor(dt_ious)    
        mean_iou = dt_ious.mean().item()
        
        # 𝑑(𝐼𝑜𝑈,𝜃_𝑔𝑡, 𝜃_𝑑𝑡)=𝐼𝑜𝑈/(1+ln(|𝜃_𝑔𝑡−𝜃_𝑑𝑡|+1))
        abs_diff_angle = torch.abs(torch.as_tensor(gt_angles) - torch.as_tensor(dt_angles))
        modulated_ious = dt_ious / (1 + torch.log(abs_diff_angle + 1))
        mean_modulated_iou = modulated_ious.mean().item()
        return precision, recall, f1_score, AP, mean_iou, mean_modulated_iou