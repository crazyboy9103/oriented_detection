import os
from typing import Dict, Any, Iterable, Tuple, Optional, Type

import torch
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont

from datasets.base import BaseDataset
from ops import boxes as box_ops

FONT_PATH = "./fonts/roboto_medium.ttf"
FONT_SIZE = 12
FONT = ImageFont.truetype(FONT_PATH, size=FONT_SIZE)
ANCHOR_TYPE = 'lt'

def plot_image(image: torch.Tensor, output: Dict[str, Any], target: Dict[str, Any], 
               data: Type[BaseDataset], o_score_threshold: float = 0.5, 
               resize: Optional[Tuple[int, int]] = None):
    _, max_h, max_w = image.shape
    image = to_pil_image(image.detach().cpu())
    draw = ImageDraw.Draw(image)

    # Draw detected objects
    if 'oboxes' in output:
        draw_detected_objects(draw, output, data, max_h, max_w, o_score_threshold)

    # Draw ground truth objects
    draw_ground_truth_objects(draw, target, data, max_h, max_w)

    # if resize is not None:
    #     image = image.resize(resize)
        
    return image, target["image_path"]

def draw_detected_objects(draw, output, data, max_h, max_w, score_threshold,):
    dt_oboxes = output['oboxes'].detach()
    dt_olabels = output['olabels'].detach()
    dt_oscores = output['oscores'].detach()
    omask = dt_oscores > score_threshold

    for dt_obox, dt_label, dt_score in zip(dt_oboxes[omask], dt_olabels[omask], dt_oscores[omask]):
        color = data.get_palette(dt_label.item())
        dt_opoly = box_ops.obb2poly(dt_obox).to(int).tolist()
        draw.polygon(dt_opoly, outline=color, width=4)
        
        str_obox = obox_to_str(dt_obox)

        text_to_draw = f'DT[{data.idx_to_class(dt_label.item())} {dt_score * 100:.2f}% {str_obox}]'
        rectangle = get_xy_bounds_text(draw, dt_opoly[:2], text_to_draw, max_h, max_w)
        draw.rectangle(rectangle, fill="black")
        draw.text(rectangle[:2], text_to_draw,
                  fill=color, font=FONT, anchor=ANCHOR_TYPE)

def draw_ground_truth_objects(draw, target, data, max_h, max_w):
    for gt_obox, gt_label in zip(target['oboxes'].detach(), target['labels'].detach()):
        color = data.get_palette(gt_label.item())
        gt_opoly = box_ops.obb2poly(gt_obox).to(int).tolist()
        draw.polygon(gt_opoly, outline=color)

        str_obox = obox_to_str(gt_obox)

        text_to_draw = f'GT[{data.idx_to_class(gt_label.item())} {str_obox}]'
        rectangle = get_xy_bounds_text(draw, gt_opoly[:2], text_to_draw, max_h, max_w)
        draw.rectangle(rectangle, fill="black")
        draw.text(rectangle[:2], text_to_draw,
                  fill=color, font=FONT, anchor=ANCHOR_TYPE)

def obox_to_str(obox: torch.Tensor) -> str:
    obox = obox.tolist()
    for i in range(4):
        obox[i] = int(obox[i])
    obox[-1] = round(obox[-1], 1)
    return str(obox)
    
def get_xy_bounds_text(draw: ImageDraw.Draw, top_left: Iterable[int], text: str, max_h, max_w, padding: int = 2) -> Tuple[int, int, int, int]:
    """
    Calculate the bounding box for the text with padding.

    Args:
        draw (ImageDraw.Draw): The drawing context.
        top_left (Iterable[int]): The top-left position to start the text.
        text (str): The text to be drawn.
        padding (int): The padding around the text.

    Returns:
        Tuple[int, int, int, int]: The bounding box for the text (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = draw.textbbox(top_left, text, font=FONT)
    return max(0, x1-padding), max(0, y1-padding), x2+padding, y2+padding