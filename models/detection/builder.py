from typing import List, Optional, Tuple, Dict, Callable
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, LastLevelMaxPool
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNConvFCHead
# Backbones
from torchvision.models.resnet import resnet18, resnet50
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torchvision.models import (
    EfficientNet, 
    efficientnet_b0, 
    efficientnet_b1, 
    efficientnet_b2, 
    efficientnet_b3, 
)
from torchvision.models.detection.backbone_utils import _mobilenet_extractor, _resnet_fpn_extractor, _validate_trainable_layers, BackboneWithFPN
# Weights
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.resnet import (
    ResNet18_Weights, 
    ResNet50_Weights, 
)
from torchvision.models.mobilenetv3 import (
    MobileNet_V3_Large_Weights
)
from torchvision.models.efficientnet import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
    EfficientNet_B3_Weights,
)

from ops.poolers import MultiScaleRotatedRoIAlign

from .roi_heads import RotatedFasterRCNNRoIHead
from .rpn import RPNHead, RotatedRegionProposalNetwork
from .transform import GeneralizedRCNNTransform
from .anchor_utils import RotatedAnchorGenerator

def _efficientnet_extractor(
    backbone: EfficientNet,
    trainable_layers: int,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[ExtraFPNBlock] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
) -> nn.Module:
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0, 2, 3, 4, 6]
    num_stages = len(stage_indices)

    # find the index of the layer from which we won't freeze
    if trainable_layers < 0 or trainable_layers > num_stages:
        raise ValueError(f"Trainable layers should be in the range [0,{num_stages}], got {trainable_layers} ")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256
    if extra_blocks is None:
        extra_blocks = LastLevelMaxPool()

    if returned_layers is None:
        returned_layers = [num_stages - 2, num_stages - 1]
    if min(returned_layers) < 0 or max(returned_layers) >= num_stages:
        raise ValueError(f"Each returned layer should be in the range [0,{num_stages - 1}], got {returned_layers} ")
    return_layers = {f"{stage_indices[k]}": str(v) for v, k in enumerate(returned_layers)}
    
    in_channels_list = []
    for i in returned_layers:
        layer = backbone[stage_indices[i]]
        if isinstance(layer, misc_nn_ops.Conv2dNormActivation):
            in_channels_list.append(layer.out_channels)

        else:
            in_channels_list.append(layer[-1].out_channels)
            
    return BackboneWithFPN(
        backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
    )

def _check_for_degenerate_boxes(targets):
    for target_idx, target in enumerate(targets):
        boxes = target["hboxes"]
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

        if self.training:
            return losses
        
        if targets:
            return losses, detections
        
        return detections

class RotatedRCNNWrapper(GeneralizedRCNN):
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
        rpn_anchor_generator=None,
        rpn=None,
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
        roi_head=None,
        box_roi_pool: nn.Module = None,
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
        
        rpn_head = rpn_head(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
        
        if box_head is None:
            resolution = box_roi_pool.output_size[0]
            box_head = TwoMLPHead(in_channels=out_channels * resolution * resolution, representation_size=1024)
        
        if box_predictor is None:
            box_predictor = OrientedRCNNPredictor(in_channels=1024, num_classes=num_classes)
            
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, **kwargs)
        
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn = rpn(
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
        
        roi_heads = roi_head(
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
        )
                
        super(RotatedRCNNWrapper, self).__init__(backbone, transform, rpn, roi_heads)

def model_builder(
    *,
    pretrained,
    pretrained_backbone,
    progress: bool = True, 
    num_classes,
    trainable_backbone_layers,
    model,
    freeze_bn,
    backbone_type,
    roi_pooler,
    **kwargs
):
    weights = None
    weights_backbone = None

    if pretrained:
        if backbone_type == "resnet50":
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.verify(weights)
        
        elif backbone_type == "resnet18":
            print("[Builder] No COCO pretrained weights for ResNet18, using ImageNet weights for backbone instead.")
            
        elif backbone_type == "mobilenetv3large":
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1	
            weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.verify(weights)
        
        elif "efficientnet" in backbone_type:
            print("[Builder] No COCO pretrained weights for EfficientNet, using ImageNet weights for backbone instead.")  

    if not weights and pretrained_backbone:
        if backbone_type == "resnet50":
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1
            weights_backbone = ResNet50_Weights.verify(weights_backbone)
        
        elif backbone_type == "resnet18":
            weights_backbone = ResNet18_Weights.IMAGENET1K_V1
            weights_backbone = ResNet18_Weights.verify(weights_backbone)
            
        elif backbone_type == "mobilenetv3large":
            weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1	
            weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)
            
        elif "efficientnet" in backbone_type:
            try:
                weights_backbone = globals().get(f"EfficientNet_{backbone_type.strip('efficientnet_').upper()}_Weights").IMAGENET1K_V1

            except AttributeError:
                raise ValueError(f"Unknown EfficientNet type {backbone_type}")
        
    is_faster_rcnn_trained = weights is not None
    is_backbone_trained = weights_backbone is not None or weights is not None
    
    fast_rcnn_norm_layer = nn.BatchNorm2d
    backbone_norm_layer = nn.BatchNorm2d

    if freeze_bn:
        if not is_faster_rcnn_trained:
            print("[Builder] WARNING: pretrained weights are not used for training, freeze_bn is ignored.")
        else:
            fast_rcnn_norm_layer = misc_nn_ops.FrozenBatchNorm2d

        if not is_backbone_trained:
            print("[Builder] WARNING: pretrained backbone weights are not used for training, freeze_bn is ignored.")
        else:
            backbone_norm_layer = misc_nn_ops.FrozenBatchNorm2d
    
    if backbone_type in ("resnet50", "resnet18"):
        max_trainable_backbone_layers = 5
    elif backbone_type == "mobilenetv3large":
        max_trainable_backbone_layers = 6 
    elif "efficientnet" in backbone_type:
        max_trainable_backbone_layers = 5
    
    trainable_backbone_layers = _validate_trainable_layers(is_backbone_trained, trainable_backbone_layers, max_value=max_trainable_backbone_layers, default_value=5)

    # Backbone
    if backbone_type == "resnet50":
        backbone = resnet50(weights=weights_backbone, progress=progress)
        backbone = _resnet_fpn_extractor(backbone, trainable_layers=trainable_backbone_layers, returned_layers=[1, 2, 3, 4], norm_layer=backbone_norm_layer)
    
    elif backbone_type == "resnet18":
        backbone = resnet18(weights=weights_backbone, progress=progress)
        backbone = _resnet_fpn_extractor(backbone, trainable_layers=trainable_backbone_layers, returned_layers=[1, 2, 3, 4], norm_layer=backbone_norm_layer)
        
    elif backbone_type == "mobilenetv3large":
        backbone = mobilenet_v3_large(weights=weights_backbone, progress=progress)	
        backbone = _mobilenet_extractor(backbone, fpn=True, trainable_layers=trainable_backbone_layers, returned_layers=[1, 2, 3, 4], norm_layer=backbone_norm_layer)	

    elif "efficientnet" in backbone_type:
        backbone = torchvision.models.__dict__[backbone_type](weights=weights_backbone, progress=progress)
        backbone = _efficientnet_extractor(backbone, trainable_layers=trainable_backbone_layers, returned_layers=[1, 2, 3, 4], norm_layer=backbone_norm_layer)
    
    # Anchors
    num_feature_maps = 5
    anchor_sizes = (
        (8, 16, 32, 64, 128, )
    ) * num_feature_maps
    aspect_ratios = ((0.1, 0.5, 1.0, 2.0,),) * num_feature_maps
    angles = ((0, 60, 120, 180, 240, 300),) * num_feature_maps
    # 90, 180, 270,
    rpn_anchor_generator = RotatedAnchorGenerator(anchor_sizes, aspect_ratios, angles) 
    
    pool_size = 7
    num_fc = 4

    box_head = FastRCNNConvFCHead(
        (backbone.out_channels, pool_size, pool_size), [256] * num_fc, [1024], norm_layer=fast_rcnn_norm_layer
    )
    
    # Roi pooler
    box_roi_pool = roi_pooler(featmap_names=["0", "1", "2", "3"], output_size=pool_size, sampling_ratio=2)
    
    model = model(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=rpn_anchor_generator,
        box_head=box_head,
        box_roi_pool=box_roi_pool,
        **kwargs,
    )

    if weights:
        model_state_dict = model.state_dict()
        trained_state_dict = weights.get_state_dict(progress=progress)
        for k, tensor in model_state_dict.items():
            trained_tensor = trained_state_dict.get(k, None)
            if trained_tensor is not None:
                if tensor.shape == trained_tensor.shape:
                    model_state_dict[k] = trained_tensor
                else:
                    print(f"[Builder] Skipped loading parameter {k} due to incompatible shapes: {tensor.shape} vs {trained_tensor.shape}")
            else:
                print(f"[Builder] Skipped loading parameter {k} which is not in the model's state dict.")

        model.load_state_dict(model_state_dict, strict=False)
        
    return model

RotatedRPNHead = partial(RPNHead, bbox_dim=5)

RotatedFasterRCNN = partial(RotatedRCNNWrapper, 
                            rpn_head=RotatedRPNHead,
                            rpn=RotatedRegionProposalNetwork, 
                            roi_head=RotatedFasterRCNNRoIHead)

rotated_faster_rcnn_builder = partial(model_builder, 
                              model=RotatedFasterRCNN, 
                              roi_pooler=MultiScaleRotatedRoIAlign)