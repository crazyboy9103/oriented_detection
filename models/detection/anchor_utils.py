from typing import List

import torch
from torch import nn, Tensor

from torchvision.models.detection.image_list import ImageList


class RotatedAnchorGenerator(nn.Module):
    """
    Compute rotated anchors used by Rotated RPN (RRPN), described in
    "Arbitrary-Oriented Scene Text Detection via Rotation Proposals".
    
    Module that generates rotated anchors for a set of feature maps and
    image sizes.

    The module support computing anchors at multiple sizes and aspect ratios
    per feature map. This module assumes aspect ratio = height / width for
    each anchor.

    sizes, aspect_ratios, angles should have the same number of elements, and it should
    correspond to the number of feature maps.

    sizes[i], aspect_ratios[i], angles[i] can have an arbitrary number of elements,
    and RotatedAnchorGenerator will output a set of sizes[i] * aspect_ratios[i] * angles[i] rotated anchors
    per spatial location for feature map i.

    Args:
        sizes (Tuple[Tuple[int]]):
        aspect_ratios (Tuple[Tuple[float]]):
        angles (Tuple[Tuple[float]]): Angles in degrees CCW, right-handed coordinate system.
    """

    __annotations__ = {
        "cell_anchors": List[torch.Tensor],
    }

    def __init__(
        self,
        sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        angles=((0, 45, 90, 135, 180, 225, 270, 315),),
    ):
        super().__init__()

        if not isinstance(sizes[0], (list, tuple)):
            # TODO change this
            sizes = tuple((s,) for s in sizes)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            aspect_ratios = (aspect_ratios,) * len(sizes)

        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self.angles = angles
        
        self.cell_anchors = [
            self.generate_anchors(size, aspect_ratio, angle) for size, aspect_ratio, angle in zip(sizes, aspect_ratios, angles)
        ]

    # TODO: https://github.com/pytorch/pytorch/issues/26792
    # For every (aspect_ratios, scales) combination, output a zero-centered anchor with those values.
    # (scales, aspect_ratios) are usually an element of zip(self.scales, self.aspect_ratios)
    # This method assumes aspect ratio = height / width for an anchor.
    def generate_anchors(
        self,
        scales: List[int],
        aspect_ratios: List[float],
        angles: List[float],
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> Tensor:
        scales = torch.as_tensor(scales, dtype=dtype, device=device)
        aspect_ratios = torch.as_tensor(aspect_ratios, dtype=dtype, device=device)
        angles = torch.as_tensor(angles, dtype=dtype, device=device)
    
        h_ratios = torch.sqrt(aspect_ratios) # sqrt(h/w)
        w_ratios = 1 / h_ratios # sqrt(w/h)

        ws = (w_ratios[:, None] * scales[None, :]).view(-1) # sqrt(w/h) * sqrt(wh) = ws
        hs = (h_ratios[:, None] * scales[None, :]).view(-1) # sqrt(h/w) * sqrt(wh) = hs

        zeros = torch.zeros_like(ws, dtype=dtype, device=device)
        anchors = torch.stack([zeros, zeros, ws, hs], dim=1)
        anchors = torch.repeat_interleave(anchors, len(angles), dim=0)
        angles = angles.repeat(len(ws)).unsqueeze(1)
        anchors_with_angles = torch.cat([anchors, angles], dim=1)
        return anchors_with_angles

    def set_cell_anchors(self, dtype: torch.dtype, device: torch.device):
        self.cell_anchors = [cell_anchor.to(dtype=dtype, device=device) for cell_anchor in self.cell_anchors]

    def num_anchors_per_location(self) -> List[int]:
        return [len(s) * len(a) * len(t) for s, a, t in zip(self.sizes, self.aspect_ratios, self.angles)]

    # For every combination of (a, (g, s), i) in (self.cell_anchors, zip(grid_sizes, strides), 0:2),
    # output g[i] anchors that are s[i] distance apart in direction i, with the same dimensions as a.
    def grid_anchors(self, grid_sizes: List[List[int]], strides: List[List[Tensor]]) -> List[Tensor]:
        anchors = []
        cell_anchors = self.cell_anchors
        torch._assert(cell_anchors is not None, "cell_anchors should not be None")
        torch._assert(
            len(grid_sizes) == len(strides) == len(cell_anchors),
            "Anchors should be Tuple[Tuple[int]] because each feature "
            "map could potentially have different sizes and aspect ratios. "
            "There needs to be a match between the number of "
            "feature maps passed and the number of sizes / aspect ratios specified.",
        )

        for size, stride, base_anchors in zip(grid_sizes, strides, cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device = base_anchors.device

            # For output anchor, compute [x_center, y_center, width, height, angle]
            shifts_x = torch.arange(0, grid_width, dtype=torch.int32, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=torch.int32, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            zeros = torch.zeros_like(shift_x, dtype=torch.int32, device=device)
            
            # For rotated anchor, required shifts for width, height, angle are 0, 0, 0. 
            shifts = torch.stack((shift_x, shift_y, zeros, zeros, zeros), dim=1)

            # For every (base anchor, output anchor) pair,
            # offset each zero-centered base anchor by the center of the output anchor.
            anchors.append((shifts.view(-1, 1, 5) + base_anchors.view(1, -1, 5)).reshape(-1, 5))

        return anchors

    def forward(self, image_list: ImageList, feature_maps: List[Tensor]) -> List[Tensor]:
        grid_sizes = [feature_map.shape[-2:] for feature_map in feature_maps]
        image_size = image_list.tensors.shape[-2:]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        strides = [
            [
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[0] // g[0]),
                torch.empty((), dtype=torch.int64, device=device).fill_(image_size[1] // g[1]),
            ]
            for g in grid_sizes
        ]
        self.set_cell_anchors(dtype, device)
        anchors_over_all_feature_maps = self.grid_anchors(grid_sizes, strides)
        anchors: List[List[torch.Tensor]] = []
        for _ in range(len(image_list.image_sizes)):
            anchors_in_image = [anchors_per_feature_map for anchors_per_feature_map in anchors_over_all_feature_maps]
            anchors.append(anchors_in_image)
        anchors = [torch.cat(anchors_per_image) for anchors_per_image in anchors]
        return anchors
    
if __name__ == "__main__":
    generator = RotatedAnchorGenerator(
        sizes = ((128),),
        aspect_ratios = ((1.0,),),
        angles = ((0, 90, 225),),
    )
    
    image_list = ImageList(torch.rand(1, 3, 512, 512), [(512, 512)])
    feature_maps = [torch.rand(1, 256, 64, 64)]
    anchors = generator(image_list, feature_maps)
    print(anchors)