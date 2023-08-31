from typing import List, Optional, Tuple, Dict, Any, Literal
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNConvFCHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
# Backbones
from torchvision.models.resnet import resnet18, resnet50, resnet101, resnet152
from torchvision.models.detection.backbone_utils import _mobilenet_extractor, _resnet_fpn_extractor, _validate_trainable_layers
# Weights
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights, 
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.resnet import (
    ResNet18_Weights, 
    ResNet50_Weights, 
    ResNet101_Weights, 
    ResNet152_Weights
)

from .roi_heads import RoIHeads as FasterRCNNRoIHeads
from .oriented_roi_heads import RoIHeads as OrientedRCNNRoIHeads
from .rpn import RPNHead, OrientedRPNHead, RegionProposalNetwork, OrientedRegionProposalNetwork
from .transform import GeneralizedRCNNTransform
from ops.poolers import MultiScaleRotatedRoIAlign

def _default_anchor_generator():
    sizes = ((4, 8, 16, 32, 64,),) * 5 
    ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    return AnchorGenerator(sizes=sizes, aspect_ratios=ratios)

def _check_for_degenerate_boxes(targets):
    for target_idx, target in enumerate(targets):
        boxes = target["bboxes"]
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
        
        oboxes = target["oboxes"]
        degenerate_oboxes = oboxes[:, 2:4] <= 0
        if degenerate_oboxes.any():
            # print the first degenerate box
            bb_idx = torch.where(degenerate_oboxes.any(dim=1))[0][0]
            degen_bb: List[float] = oboxes[bb_idx].tolist()
            torch._assert(
                False,
                "All bounding boxes should have positive height and width."
                f" Found invalid box {degen_bb} for target at index {target_idx}.",
            )

class RotatedFastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box + angle regression layers 
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.obbox_pred = nn.Linear(in_channels, num_classes * 5)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        obbox_deltas = self.obbox_pred(x)
        return scores, bbox_deltas, obbox_deltas

class OrientedRCNNPredictor(nn.Module):
    """
    Standard classification + Oriented bounding box regression layers 
    for Oriented R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.obbox_pred = nn.Linear(in_channels, num_classes * 5)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        obbox_deltas = self.obbox_pred(x)
        return scores, obbox_deltas
    
class GeneralizedRCNN(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module,                
        transform: nn.Module,
        rpn: nn.Module,
        roi_heads: nn.Module,
    ) -> None:
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.transform = transform
        self.rpn = rpn
        self.roi_heads = roi_heads

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
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        if targets is not None:
            _check_for_degenerate_boxes(targets)

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
            return losses, detections
        else:
            if self.training:
                return losses
            
            if targets:
                return losses, detections
            
            return detections
        
class RotatedFasterRCNN(GeneralizedRCNN):
    def __init__(
        self, 
        backbone: nn.Module,                
        num_classes: int = 16,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator: nn.Module = _default_anchor_generator(),
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool: nn.Module = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=0),
        box_head: nn.Module = None,
        box_predictor: nn.Module = None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ) -> None:
        out_channels = backbone.out_channels
        
        if rpn_head is None:
            rpn_head = RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            box_head = TwoMLPHead(in_channels=out_channels * resolution ** 2, representation_size=1024)
        
        if box_predictor is None:
            box_predictor = RotatedFastRCNNPredictor(in_channels=1024, num_classes=num_classes)
            
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )
        
        roi_heads = FasterRCNNRoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            nms_thresh_rotated = 0.1
        )
                
        super(RotatedFasterRCNN, self).__init__(backbone, transform, rpn, roi_heads)

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
                    boxes = target["bboxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
                        
                    oboxes = target["oboxes"]
                    if isinstance(oboxes, torch.Tensor):
                        torch._assert(
                            len(oboxes.shape) == 2 and oboxes.shape[-1] == 5,
                            f"Expected target boxes to be a tensor of shape [N, 5], got {oboxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(oboxes)}.")

        return super(RotatedFasterRCNN, self).forward(images, targets)

 
class OrientedRCNN(GeneralizedRCNN):
    def __init__(
        self, 
        backbone: nn.Module,                
        num_classes: int = 16,
        # transform parameters
        min_size=800,
        max_size=1333,
        image_mean=None,
        image_std=None,
        # RPN parameters
        rpn_anchor_generator: nn.Module = _default_anchor_generator(),
        rpn_head=None,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
        rpn_score_thresh=0.0,
        # Box parameters
        box_roi_pool: nn.Module = MultiScaleRotatedRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=0),
        box_head: nn.Module = None,
        box_predictor: nn.Module = None,
        box_score_thresh=0.05,
        box_nms_thresh=0.5,
        box_detections_per_img=100,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.5,
        box_batch_size_per_image=512,
        box_positive_fraction=0.25,
        bbox_reg_weights=None,
        **kwargs,
    ) -> None:
        out_channels = backbone.out_channels
        
        if rpn_head is None:
            rpn_head = OrientedRPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            box_head = TwoMLPHead(in_channels=out_channels * resolution ** 2, representation_size=1024)
        
        if box_predictor is None:
            box_predictor = OrientedRCNNPredictor(in_channels=1024, num_classes=num_classes)
            
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = OrientedRegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh,
        )
        
        roi_heads = OrientedRCNNRoIHeads(
            # Box
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
            nms_thresh_rotated = 0.1
        )
                
        super(OrientedRCNN, self).__init__(backbone, transform, rpn, roi_heads)

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
                    oboxes = target["oboxes"]
                    if isinstance(oboxes, torch.Tensor):
                        torch._assert(
                            len(oboxes.shape) == 2 and oboxes.shape[-1] == 5,
                            f"Expected target boxes to be a tensor of shape [N, 5], got {oboxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(oboxes)}.")

        return super(OrientedRCNN, self).forward(images, targets)


def model_builder(
    pretrained: bool = True,
    pretrained_backbone: bool = True,
    progress: bool = True, 
    num_classes: Optional[int] = 91,
    trainable_backbone_layers: Optional[int] = None,
    version: Literal[1, 2] = 1,
    rpn_head: Optional[nn.Module] = None,
    model: Optional[nn.Module] = None,
    **kwargs
):
    if version == 1 and pretrained:
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        weights = FasterRCNN_ResNet50_FPN_Weights.verify(weights)
    elif version == 2 and pretrained:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.verify(weights)
    else:
        weights = None
        
    if weights and num_classes is None:
        num_classes = len(weights.meta["categories"])
        
    weights_backbone = ResNet50_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    weights_backbone = ResNet50_Weights.verify(weights_backbone)
    
    
    is_backbone_trained = weights_backbone is not None
    is_faster_rcnn_trained = weights is not None
    
    backbone_norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_backbone_trained else nn.BatchNorm2d
    fast_rcnn_norm_layer = misc_nn_ops.FrozenBatchNorm2d if is_faster_rcnn_trained else nn.BatchNorm2d
    
    trainable_backbone_layers = _validate_trainable_layers(is_backbone_trained, trainable_backbone_layers, max_value=5, default_value=3)

    backbone = resnet50(weights=weights_backbone, progress=progress)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers, [1, 2, 3, 4], norm_layer=backbone_norm_layer)
    
    rpn_anchor_generator = _default_anchor_generator()
    
    if version == 1:
        rpn_head = None
        box_head = None
        
    elif version == 2:
        rpn_head = rpn_head(backbone.out_channels, rpn_anchor_generator.num_anchors_per_location()[0], conv_depth=2)
        box_head = FastRCNNConvFCHead(
            (backbone.out_channels, 7, 7), [256, 256, 256, 256], [1024], norm_layer=fast_rcnn_norm_layer
        )
    
    model = model(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        rpn_head=rpn_head,
        box_head=box_head,
        **kwargs,
    )

    if weights:
        model_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        trained_state_dict = weights.get_state_dict(progress=progress)
        for k, tensor in model_state_dict.items():
            trained_tensor = trained_state_dict.get(k, None)
            if trained_tensor is not None:
                if tensor.shape == trained_tensor.shape:
                    model_state_dict[k] = trained_tensor
                else:
                    print(f"Skipped loading parameter {k} due to incompatible shapes: {tensor.shape} vs {trained_tensor.shape}")
            else:
                print(f"Skipped loading parameter {k} which is not in the model's state dict.")

        model.load_state_dict(model_state_dict, strict=False)
        
        if version == 1 and weights == FasterRCNN_ResNet50_FPN_Weights.COCO_V1:
            for module in model.modules():
                if isinstance(module, misc_nn_ops.FrozenBatchNorm2d):
                    module.eps = 0.0
            
    return model

faster_rcnn_builder = partial(model_builder, rpn_head=RPNHead, model=RotatedFasterRCNN)
oriented_rcnn_builder = partial(model_builder, rpn_head=OrientedRPNHead, model=OrientedRCNN)