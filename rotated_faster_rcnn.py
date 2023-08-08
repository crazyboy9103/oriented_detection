from typing import List, Optional, Tuple, Dict, Union, Iterator, Callable, Any
from collections import OrderedDict
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from pytorch_lightning import LightningModule
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from pytorch_lightning.metrics.functional import accuracy
from dota_dataset import DotaDataset

class RotatedFasterRCNN(LightningModule):
    def __init__(self):
        super(RotatedFasterRCNN, self).__init__()
        self.model = _RotatedFasterRCNN
        self.num_classes = 10  # Number of classes in the dataset
        self.lr = 0.001

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = [F.to_tensor(img) for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [F.to_tensor(img) for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        outputs = self.model(images)
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        pred_boxes = outputs[0]['boxes']
        pred_labels = outputs[0]['labels']
        target_boxes = targets[0]['boxes']
        target_labels = targets[0]['labels']

        acc = accuracy(pred_labels, target_labels)

        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer

    def train_dataloader(self):
        train_dataset = DotaDataset(...)  # Initialize the DotaDataset with appropriate parameters
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        return train_loader

    def val_dataloader(self):
        val_dataset = DotaDataset(...)  # Initialize the DotaDataset with appropriate parameters
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
        return val_loader


class _RotatedFasterRCNN(GeneralizedRCNN):
    def __init__(self, backbone: nn.Module, rpn: nn.Module, roi_heads: nn.Module, transform: nn.Module) -> None:
        super(_RotatedFasterRCNN, self).__init__(backbone, rpn, roi_heads, transform)

    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 5,
                            f"Expected target boxes to be a tensor of shape [N, 5], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)    