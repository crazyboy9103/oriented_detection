from torchvision.ops._register_onnx_ops import _register_custom_op
from .boxes import (
    # batched_nms,
    # box_area,
    # box_convert,
    # box_iou,
    # clip_boxes_to_image,
    # complete_box_iou,
    # distance_box_iou,
    # generalized_box_iou,
    # masks_to_boxes,
    # nms,
    # remove_small_boxes,
    # TODO !!
    box_iou
)
# from torchvision.ops.ciou_loss import complete_box_iou_loss
# from torchvision.ops.deform_conv import deform_conv2d, DeformConv2d
# from torchvision.ops.diou_loss import distance_box_iou_loss
# from torchvision.ops.drop_block import drop_block2d, drop_block3d, DropBlock2d, DropBlock3d
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.focal_loss import sigmoid_focal_loss
# from torchvision.ops.giou_loss import generalized_box_iou_loss
from torchvision.ops.misc import Conv2dNormActivation, FrozenBatchNorm2d, MLP, Permute, SqueezeExcitation #, Conv3dNormActivation
from torchvision.ops.poolers import MultiScaleRoIAlign
# from torchvision.ops.ps_roi_align import ps_roi_align, PSRoIAlign
# from torchvision.ops.ps_roi_pool import ps_roi_pool, PSRoIPool
from torchvision.ops.roi_align import roi_align, RoIAlign
from torchvision.ops.roi_pool import roi_pool, RoIPool
from torchvision.ops.stochastic_depth import stochastic_depth, StochasticDepth

_register_custom_op()


__all__ = [
    # TODO !!
    # "masks_to_boxes",
    # "deform_conv2d",
    # "DeformConv2d",
    # "nms",
    # "batched_nms",
    # "remove_small_boxes",
    # "clip_boxes_to_image",
    # "box_convert",
    # "box_area",
    # "box_iou",
    # "generalized_box_iou",
    # "distance_box_iou",
    # "complete_box_iou",
    # "roi_align",
    # "RoIAlign",
    # "roi_pool",
    # "RoIPool",
    # "ps_roi_align",
    # "PSRoIAlign",
    # "ps_roi_pool",
    # "PSRoIPool",
    # "MultiScaleRoIAlign",
    # "FeaturePyramidNetwork",
    # "sigmoid_focal_loss",
    # "stochastic_depth",
    # "StochasticDepth",
    # "FrozenBatchNorm2d",
    # "Conv2dNormActivation",
    # "Conv3dNormActivation",
    # "SqueezeExcitation",
    # "MLP",
    # "Permute",
    # "generalized_box_iou_loss",
    # "distance_box_iou_loss",
    # "complete_box_iou_loss",
    # "drop_block2d",
    # "DropBlock2d",
    # "drop_block3d",
    # "DropBlock3d",
]
