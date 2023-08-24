# Adapted from detectron2.layers.roi_align_rotated
# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair
import torchvision
from torchvision.utils import _log_api_usage_once
from torchvision.ops import roi_align
from torchvision.ops.poolers import _onnx_merge_levels, _convert_to_roi_format, _infer_scale

from detectron2._C import roi_align_rotated_backward, roi_align_rotated_forward

class _ROIAlignRotated(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = roi_align_rotated_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (rois,) = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = roi_align_rotated_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None, None


roi_align_rotated = _ROIAlignRotated.apply


class ROIAlignRotated(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.

        Note:
            ROIAlignRotated supports continuous coordinate by default:
            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5).
        """
        super(ROIAlignRotated, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio


    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx6 boxes. First column is the index into N.
                The other 5 columns are (x_ctr, y_ctr, width, height, angle_degrees).
        """
        assert rois.dim() == 2 and rois.size(1) == 6
        orig_dtype = input.dtype
        if orig_dtype == torch.float16:
            input = input.float()
            rois = rois.float()
        output_size = _pair(self.output_size)

        # Scripting for Autograd is currently unsupported.
        # This is a quick fix without having to rewrite code on the C++ side
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return roi_align_rotated_forward(
                input, rois, self.spatial_scale, output_size[0], output_size[1], self.sampling_ratio
            ).to(dtype=orig_dtype)

        return roi_align_rotated(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        ).to(dtype=orig_dtype)


    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr


# Adapted from torchvision.ops.poolers

def box_area(box: Tensor):
    """
    Args:
        box (Tensor[N, 5]): boxes in ``(cx, cy, w, h, a)`` format
    Returns:
        area (Tensor[N]): area for each box
    """
    return box[:, 2] * box[:, 3]

class LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        Args:
            boxlists (list[Tensor[N, 5]]): A list of N tensors
        """
        # Compute level ids
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


@torch.fx.wrap
def _setup_scales(
    features: List[Tensor], image_shapes: List[Tuple[int, int]], canonical_scale: int, canonical_level: int
) -> Tuple[List[float], LevelMapper]:
    if not image_shapes:
        raise ValueError("images list should not be empty")
    max_x = 0
    max_y = 0
    for shape in image_shapes:
        max_x = max(shape[0], max_x)
        max_y = max(shape[1], max_y)
    original_input_shape = (max_x, max_y)

    scales = [_infer_scale(feat, original_input_shape) for feat in features]
    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
    lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()

    map_levels = LevelMapper(
        int(lvl_min),
        int(lvl_max),
        canonical_scale=canonical_scale,
        canonical_level=canonical_level,
    )
    return scales, map_levels


@torch.fx.wrap
def _filter_input(x: Dict[str, Tensor], featmap_names: List[str]) -> List[Tensor]:
    x_filtered = []
    for k, v in x.items():
        if k in featmap_names:
            x_filtered.append(v)
    return x_filtered


@torch.fx.wrap
def _multiscale_rotated_roi_align(
    x_filtered: List[Tensor],
    boxes: List[Tensor],
    output_size: List[int],
    sampling_ratio: int,
    scales: Optional[List[float]],
    mapper: Optional[LevelMapper],
) -> Tensor:
    """
    Args:
        x_filtered (List[Tensor]): List of input tensors.
        boxes (List[Tensor[N, 5]]): boxes to be used to perform the pooling operation, in
            (cx, cy, w, h, a) format and in the image reference size, not the feature map
            reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        output_size (Union[List[Tuple[int, int]], List[int]]): size of the output
        sampling_ratio (int): sampling ratio for ROIAlign
        scales (Optional[List[float]]): If None, scales will be automatically inferred. Default value is None.
        mapper (Optional[LevelMapper]): If none, mapper will be automatically inferred. Default value is None.
    Returns:
        result (Tensor)
    """
    if scales is None or mapper is None:
        raise ValueError("scales and mapper should not be None")

    num_levels = len(x_filtered)
    rois = _convert_to_roi_format(boxes)

    if num_levels == 1:
        return roi_align_rotated(
            x_filtered[0],
            rois,
            output_size,
            scales[0],
            sampling_ratio,
        )

    levels = mapper(boxes)

    num_rois = len(rois)
    num_channels = x_filtered[0].shape[1]

    dtype, device = x_filtered[0].dtype, x_filtered[0].device
    result = torch.zeros(
        (
            num_rois,
            num_channels,
        )
        + output_size,
        dtype=dtype,
        device=device,
    )

    tracing_results = []
    for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
        idx_in_level = torch.where(levels == level)[0]
        rois_per_level = rois[idx_in_level]

        result_idx_in_level = roi_align_rotated(
            per_level_feature,
            rois_per_level,
            output_size,
            scale,
            sampling_ratio,
        )

        if torchvision._is_tracing():
            tracing_results.append(result_idx_in_level.to(dtype))
        else:
            # result and result_idx_in_level's dtypes are based on dtypes of different
            # elements in x_filtered.  x_filtered contains tensors output by different
            # layers.  When autocast is active, it may choose different dtypes for
            # different layers' outputs.  Therefore, we defensively match result's dtype
            # before copying elements from result_idx_in_level in the following op.
            # We need to cast manually (can't rely on autocast to cast for us) because
            # the op acts on result in-place, and autocast only affects out-of-place ops.
            result[idx_in_level] = result_idx_in_level.to(result.dtype)

    if torchvision._is_tracing():
        result = _onnx_merge_levels(levels, tracing_results)

    return result


class MultiScaleRotatedRoIAlign(nn.Module):
    """
    Multi-scale Rotated RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics specified in eq. 1
    of the `Feature Pyramid Network paper <https://arxiv.org/abs/1612.03144>`_.
    They keyword-only parameters ``canonical_scale`` and ``canonical_level``
    correspond respectively to ``224`` and ``k0=4`` in eq. 1, and
    have the following meaning: ``canonical_level`` is the target level of the pyramid from
    which to pool a region of interest with ``w x h = canonical_scale x canonical_scale``.

    Args:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
        canonical_scale (int, optional): canonical_scale for LevelMapper
        canonical_level (int, optional): canonical_level for LevelMapper

    Examples::

        >>> m = MultiScaleRotatedRoIAlign(['feat1', 'feat3'], 3, 2)
        >>> i = OrderedDict()
        >>> i['feat1'] = torch.rand(1, 5, 64, 64)
        >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        >>> i['feat3'] = torch.rand(1, 5, 16, 16)
        >>> # create some random rotated bounding boxes
        >>> boxes = torch.rand(6, 5) * 256; boxes[:, 2:4] += boxes[:, :2]
        >>> # original image size, before computing the feature maps
        >>> image_sizes = [(512, 512)]
        >>> output = m(i, [boxes], image_sizes)
        >>> print(output.shape)
        >>> torch.Size([6, 5, 3, 3])

    """

    __annotations__ = {"scales": Optional[List[float]], "map_levels": Optional[LevelMapper]}

    def __init__(
        self,
        featmap_names: List[str],
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
        *,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ):
        super().__init__()
        _log_api_usage_once(self)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level

    def forward(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 5]]): boxes to be used to perform the pooling operation, in
                (cx, cy, w, h, a) format and in the image reference size, not the feature map
                reference. The coordinate must satisfy ``0 <= cx, w < W`` and ``0 <= cy, h < H and ``.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        x_filtered = _filter_input(x, self.featmap_names)
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(
                x_filtered, image_shapes, self.canonical_scale, self.canonical_level
            )

        return _multiscale_rotated_roi_align(
            x_filtered,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
            f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )