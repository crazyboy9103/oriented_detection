from typing import List, Tuple

from torch.nn import functional as F
from torch import nn, Tensor
import torch

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def rotated_fastrcnn_loss(class_logits, hbox_regression, obox_regression, 
                          labels, hbox_regression_targets, obox_regression_targets):
    # type: (Tensor, Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor, Tensor]
    """
    Computes the loss for Rotated Faster R-CNN.

    Args:
        class_logits (Tensor)
        hbox_regression (Tensor)
        obox_regression (Tensor)
        labels (list[Tensor])
        hbox_regression_targets (Tensor)
        obox_regression_targets (Tensor)
        
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
        obox_loss (Tensor)
    """
   
    N, num_classes = class_logits.shape
    
    labels = torch.cat(labels, dim=0)
    try:
        # classification_loss = F.cross_entropy(class_logits, labels)
        classification_loss = FocalLoss()(class_logits, labels)
    except:
        classification_loss = torch.tensor(0.0)
        
    pos_inds = torch.where(labels > 0)[0]
    labels_pos = labels[pos_inds]
    hbox_regression_targets = torch.cat(hbox_regression_targets, dim=0)
    obox_regression_targets = torch.cat(obox_regression_targets, dim=0)
    
    def compute_box_loss(regression, regression_targets, horizontal=True):
        # get indices that correspond to the regression targets for
        # the corresponding ground truth labels, to be used with
        # advanced indexing
        box_dim = 4 if horizontal else 5
        regression = regression.reshape(N, regression.size(-1) // box_dim, box_dim)
        box_loss = F.smooth_l1_loss(
            regression[pos_inds, labels_pos],
            regression_targets[pos_inds],
            beta=1.0 / 9,
            reduction="mean",
        )
        # box_loss = box_loss / labels.numel()
        return box_loss
    # Compute for horizontal branch 
    hbox_loss = compute_box_loss(hbox_regression, hbox_regression_targets, horizontal=True)
    # Compute for rotated branch
    obox_loss = compute_box_loss(obox_regression, obox_regression_targets, horizontal=False)
    return classification_loss, hbox_loss, obox_loss

def oriented_rcnn_loss(class_logits, obox_regression, labels, obox_regression_targets):
    # type: (Tensor, Tensor, List[Tensor], Tensor) -> Tuple[Tensor, Tensor, Tensor]
    """
    Computes the loss for Oriented R-CNN.

    Args:
        class_logits (Tensor)
        obox_regression (Tensor)
        labels (list[Tensor])
        obox_regression_targets (Tensor)
        
    Returns:
        classification_loss (Tensor)
        obox_loss (Tensor)
    """
   
    N, num_classes = class_logits.shape
    
    labels = torch.cat(labels, dim=0)
    try:
        # classification_loss = F.cross_entropy(class_logits, labels)
        classification_loss = FocalLoss()(class_logits, labels)
    except:
        classification_loss = torch.tensor(0.0)
        
    pos_inds = torch.where(labels > 0)[0]
    labels_pos = labels[pos_inds]
    obox_regression_targets = torch.cat(obox_regression_targets, dim=0)
    obox_regression = obox_regression.reshape(N, obox_regression.size(-1) // 5, 5)
    obox_loss = F.smooth_l1_loss(
        obox_regression[pos_inds, labels_pos],
        obox_regression_targets[pos_inds],
        beta=1.0 / 9,
        reduction="mean",
    )
    # box_loss = box_loss / labels.numel()
    return classification_loss, obox_loss