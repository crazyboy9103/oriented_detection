import os
from typing import Dict, Any, Iterable

import cv2
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont

from ops import boxes as box_ops

FONT = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans-Bold.ttf')
FONT = ImageFont.truetype(FONT, size=8)
ANCHOR_TYPE = 'lt'

def plot_image(image: torch.Tensor, output: Dict[str, Any], target: Dict[str, Any], data, o_score_threshold: float = 0.3):
    image = to_pil_image(image.detach().cpu())
    draw = ImageDraw.Draw(image)

    if 'oboxes' in output:
        dt_oboxes = output['oboxes'].detach()
        dt_olabels = output['olabels'].detach()
        dt_oscores = output['oscores'].detach()
        omask = dt_oscores > o_score_threshold

        dt_oboxes = dt_oboxes[omask].cpu()
        dt_opolys = box_ops.obb2poly(dt_oboxes).to(int).tolist()
        dt_olabels = dt_olabels[omask].cpu().tolist()
        dt_oscores = dt_oscores[omask].cpu().tolist()

        for dt_opoly, dt_label, dt_score in zip(dt_opolys, dt_olabels, dt_oscores):
            color = data.get_palette(dt_label)
            dt_label = data.idx_to_class(dt_label)
            draw.polygon(dt_opoly, outline=color, width=5)
            text_to_draw = f'{dt_label} {dt_score:.2f}'
            rectangle = get_xy_bounds_text(draw, dt_opoly[:2], text_to_draw)
            draw.rectangle(rectangle, fill="black")
            draw.text([rectangle[0], (rectangle[1] + rectangle[3]) // 2], text_to_draw,
                      fill=color, font=FONT, anchor=ANCHOR_TYPE)

    gt_boxes = target['bboxes'].detach().cpu().tolist()
    gt_opolys = target['polygons'].detach().cpu().tolist()
    gt_labels = target['labels'].detach().cpu().tolist()

    # gts
    for gt_box, gt_opoly, gt_label in zip(gt_boxes, gt_opolys, gt_labels):
        color = data.get_palette(gt_label)
        gt_label = data.idx_to_class(gt_label)
        draw.rectangle(gt_box, outline=color)
        draw.polygon(gt_opoly, outline=color)
        text_to_draw = f'GT {gt_label}'
        rectangle = get_xy_bounds_text(draw, gt_box[:2], text_to_draw)
        draw.rectangle(rectangle, fill="black")
        draw.text([rectangle[0], (rectangle[1] + rectangle[3]) // 2], text_to_draw,
                  fill=color, font=FONT, anchor=ANCHOR_TYPE)

    return image, target["image_path"]


def get_xy_bounds_text(draw: ImageDraw.Draw, top_left: Iterable, text: str, padding:int =5):
    top_left = top_left[:]
    top_left[0] = max(0, top_left[0]-padding)
    top_left[1] = max(0, top_left[1]-padding)
    
    x1, y1, x2, y2 = draw.textbbox(
        xy=top_left, text=text, font=FONT, anchor=ANCHOR_TYPE)
    return x1, y1, x2 + padding, y2 + padding