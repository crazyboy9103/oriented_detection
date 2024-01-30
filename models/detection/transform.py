import math
from typing import Any, Dict, List, Optional, Tuple, Literal

import torch
from torch import nn, Tensor
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchvision.models.detection.image_list import ImageList

def _resize_boxes(boxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

def _resize_oboxes(oboxes: Tensor, original_size: List[int], new_size: List[int]) -> Tensor:
    # Important: oboxes are (cx, cy, w, h, a)
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=oboxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=oboxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    cx, cy, w, h, a = oboxes.unbind(1)

    # Adjust angle
    # Converting angle from degrees to radians for calculation
    a_rad = torch.deg2rad(a)
    # Adjust the angle based on the change in aspect ratio
    sin_a = torch.sin(a_rad)
    cos_a = torch.cos(a_rad)
    
    a_adjusted_rad = torch.atan2(sin_a * ratio_height, cos_a * ratio_width)
    # Converting angle back to degrees
    a_adjusted = torch.rad2deg(a_adjusted_rad)
    a_adjusted = torch.where(a_adjusted < 0, a_adjusted + 360.0, a_adjusted)

    cx = cx * ratio_width
    cy = cy * ratio_height
    w = w * ratio_width
    h = h * ratio_height
    output = torch.stack((cx, cy, w, h, a_adjusted), dim=1)
    return output

def _flip_boxes(boxes: Tensor, new_size: List[int], direction: Literal['horizontal', 'vertical', 'diagonal']) -> Tensor:
    orig_shape = boxes.shape
    boxes = boxes.reshape((-1, 4))
    flipped = boxes.clone()
    height, width = new_size
    
    if direction == 'horizontal':
        flipped[..., 0::4] = width - boxes[..., 2::4]
        flipped[..., 2::4] = width - boxes[..., 0::4]
    elif direction == 'vertical':
        flipped[..., 1::4] = height - boxes[..., 3::4]
        flipped[..., 3::4] = height - boxes[..., 1::4]
    elif direction == 'diagonal':
        flipped[..., 0::4] = width - boxes[..., 2::4]
        flipped[..., 1::4] = height - boxes[..., 3::4]
        flipped[..., 2::4] = width - boxes[..., 0::4]
        flipped[..., 3::4] = height - boxes[..., 1::4]
    else:
        raise ValueError(f'Invalid flipping direction "{direction}"')
    
    return flipped.reshape(orig_shape)

def _flip_oboxes(oboxes: Tensor, new_size: List[int], direction: Literal['horizontal', 'vertical', 'diagonal']) -> Tensor:
    orig_shape = oboxes.shape
    oboxes = oboxes.reshape((-1, 5))
    flipped = oboxes.clone()
    height, width = new_size
    
    if direction == 'horizontal':
        flipped[:, 0] = width - oboxes[:, 0] - 1
        # Adjust angle for horizontal flip
        flipped[:, -1] = (180 - oboxes[:, -1]) % 360

    elif direction == 'vertical':
        flipped[:, 1] = height - oboxes[:, 1] - 1
        # Adjust angle for vertical flip
        flipped[:, -1] = (360 - oboxes[:, -1]) % 360

    elif direction == 'diagonal':
        flipped[:, 0] = width - oboxes[:, 0] - 1
        flipped[:, 1] = height - oboxes[:, 1] - 1
        # Adjust angle for diagonal flip
        # This combines both horizontal and vertical flips
        flipped[:, -1] = (180 - flipped[:, -1]) % 360
        flipped[:, -1] = (360 - flipped[:, -1]) % 360
    else:
        raise ValueError(f'Invalid flipping direction "{direction}"')
    
    return flipped.reshape(orig_shape)

def _resize_image(
    image: Tensor,
    self_min_size: float,
    self_max_size: float,
    target: Optional[Dict[str, Tensor]] = None,
    fixed_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
    
    im_shape = torch.tensor(image.shape[-2:])

    size: Optional[List[int]] = None
    scale_factor: Optional[float] = None
    recompute_scale_factor: Optional[bool] = None
    if fixed_size is not None:
        size = [fixed_size[1], fixed_size[0]]
    else:
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(self_min_size / min_size, self_max_size / max_size)

        scale_factor = scale.item()
        recompute_scale_factor = True

    image = torch.nn.functional.interpolate(
        image[None],
        size=size,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=recompute_scale_factor,
        align_corners=False,
    )[0]

    if target is None:
        return image, target
    
    return image, target


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it performs are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(
        self,
        min_size: int,
        max_size: int,
        image_mean: List[float],
        image_std: List[float],
        size_divisible: int = 32,
        fixed_size: Optional[Tuple[int, int]] = None,
        **kwargs: Any,
    ):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = size_divisible
        self.fixed_size = fixed_size
        self._skip_resize = kwargs.pop("_skip_resize", False)
        self._skip_flip = kwargs.pop("_skip_flip", False)
        self._skip_image_transform = kwargs.pop("_skip_image_transform", False)

    def forward(
        self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[ImageList, Optional[List[Dict[str, Tensor]]]]:
        images = [img for img in images]
        if targets is not None:
            # make a copy of targets to avoid modifying it in-place
            # once torchscript supports dict comprehension
            # this can be simplified as follows
            # targets = [{k: v for k,v in t.items()} for t in targets]
            targets_copy: List[Dict[str, Tensor]] = []
            for t in targets:
                data: Dict[str, Tensor] = {}
                for k, v in t.items():
                    data[k] = v
                targets_copy.append(data)
            targets = targets_copy
        
        image_transform = self.get_image_transform(self.training)
        for i, image in enumerate(images):
            target = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(f"images is expected to be a list of 3d tensors of shape [C, H, W], got {image.shape}")
            
            if not self._skip_image_transform and image_transform:
                image = image_transform(image)
           
            image, target = self.resize(image, target)
            
            if not self._skip_flip and self.training:
                image, target = self.flip_bboxes(image, target)
            
            image = self.normalize(image)
            
            images[i] = image
            if targets is not None and target is not None:
                targets[i] = target

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images, size_divisible=self.size_divisible)
        image_sizes_list: List[Tuple[int, int]] = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image: Tensor) -> Tensor:
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def torch_choice(self, k: List[int]) -> int:
        """
        Implements `random.choice` via torch ops, so it can be compiled with
        TorchScript. Remove if https://github.com/pytorch/pytorch/issues/25803
        is fixed.
        """
        index = int(torch.empty(1).uniform_(0.0, float(len(k))).item())
        return k[index]

    def resize(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        h, w = image.shape[-2:]
        if self.training:
            if self._skip_resize:
                return image, target
            size = float(self.torch_choice(self.min_size))
        else:
            # FIXME assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        image, target = _resize_image(image, size, float(self.max_size), target, self.fixed_size)

        if target is None:
            return image, target

        if "hboxes" in target:
            target["hboxes"] = _resize_boxes(target["hboxes"], (h, w), image.shape[-2:])
        
        if "oboxes" in target:
            target["oboxes"] = _resize_oboxes(target["oboxes"], (h, w), image.shape[-2:])
        return image, target
    
    def flip_bboxes(
        self,
        image: Tensor,
        target: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        if self._skip_flip or self.training is False or target is None:
            return image, target
            
        direction = self.torch_choice(['horizontal', 'vertical', 'diagonal'])
        
        if direction == 'horizontal':
            image = TF.hflip(image)
            
        elif direction == 'vertical':
            image = TF.vflip(image)
             
        elif direction == 'diagonal':
            image = TF.vflip(TF.hflip(image))
        
        if "hboxes" in target:
            target["hboxes"] = _flip_boxes(target["hboxes"], image.shape[-2:], direction)
        if "oboxes" in target:
            target["oboxes"] = _flip_oboxes(target["oboxes"], image.shape[-2:], direction)
        return image, target

    def max_by_axis(self, the_list: List[List[int]]) -> List[int]:
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images: List[Tensor], size_divisible: int = 32) -> Tensor:
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(
        self,
        result: List[Dict[str, Tensor]],
        image_shapes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
    ) -> List[Dict[str, Tensor]]:
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            if "hboxes" in result[i]:
                result[i]["hboxes"] = _resize_boxes(pred["hboxes"], im_s, o_im_s)
            
            if "oboxes" in result[i]:
                result[i]["oboxes"] = _resize_oboxes(pred["oboxes"], im_s, o_im_s)
        return result
    
    @staticmethod
    def get_image_transform(train: bool) -> T.Compose:
        return T.Compose([
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.2,
            ),
            T.GaussianBlur(3, sigma=(0.1, 2.0)),
            T.RandomGrayscale(p=0.1),
        ]) if train else None
        
    def __repr__(self) -> str:
        format_string = f"{self.__class__.__name__}("
        _indent = "\n    "
        format_string += f"{_indent}Normalize(mean={self.image_mean}, std={self.image_std})"
        format_string += f"{_indent}Resize(min_size={self.min_size}, max_size={self.max_size}, mode='bilinear')"
        format_string += "\n)"
        return format_string


