from typing import List, Tuple

from torch.nn import functional as F
from torch import nn, Tensor
import torch
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight # weight parameter will act as the alpha parameter to balance class weights
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def rotated_faster_rcnn_loss(class_logits, obox_regression, labels, obox_regression_targets):
    # type: (Tensor, Tensor, List[Tensor], Tensor) -> Tuple[Tensor, Tensor, Tensor]
    """
    Computes the loss for Oriented R-CNN.

    Args:
        class_logits (Tensor) : N x C
        obox_regression (Tensor) : (N x C x 6)
        labels (list[Tensor])
        obox_regression_targets (Tensor)
        
    Returns:
        classification_loss (Tensor)
        obox_loss (Tensor)
    """
   
    N, num_classes = class_logits.shape
    
    labels = torch.cat(labels, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)
    # classification_loss = FocalLoss()(class_logits, labels)
        
    pos_inds = (labels > 0).nonzero().squeeze()
    labels_pos = labels[pos_inds]
    
    obox_regression_targets = torch.cat(obox_regression_targets, dim=0)
    obox_regression = obox_regression.reshape(N, obox_regression.size(-1) // 6, 6) # N x C x 5
    
    preds = obox_regression[pos_inds, labels_pos]
    targets = obox_regression_targets[pos_inds]
    
    preds_coords = preds[:, :4]
    preds_angles = preds[:, 4:]

    targets_coords = targets[:, :4]
    targets_angles = targets[:, 4:]

    obox_loss = F.smooth_l1_loss(
        preds_coords, 
        targets_coords,
        beta=1.0 / 9,
        reduction="sum",
    ) / labels.numel()

    angle_loss = F.smooth_l1_loss(
        preds_angles, 
        targets_angles,
        beta=1.0,
        reduction="sum",
    ) / labels.numel()
    return classification_loss, obox_loss, angle_loss