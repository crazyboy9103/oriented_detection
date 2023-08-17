import os
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import wandb
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw, ImageFont
import cv2

from models.detection.rotated_faster_rcnn import rotated_fasterrcnn_resnet50_fpn_v2
from scheduler import CosineAnnealingWarmUpRestartsDecay
from ops.boxes import obb2poly
from datasets.mvtec import MVTecDataset
from datasets.dota import DotaDataset
from evaluation.evaluator import eval_rbbox_map

FONT = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
FONT = ImageFont.truetype(FONT, size=8)
ANCHOR_TYPE = 'lt'

# TODO add more args


@dataclass
class MvtecDataConfig:
    image_mean: Tuple[float, float, float] = (
        211.35 / 255, 166.559 / 255, 97.271 / 255)
    image_std: Tuple[float, float, float] = (
        43.849 / 255, 40.172 / 255, 30.459 / 255)


@dataclass
class TrainConfig:
    pretrained: bool = True
    pretrained_backbone: bool = True
    progress: bool = True
    num_classes: Optional[int] = 13 + 1
    trainable_backbone_layers: Optional[int] = 4


@dataclass
class RotatedFasterRCNNConfig:
    # transform parameters
    min_size: int = 512
    max_size: int = 512
    image_mean: Optional[Tuple[float, float, float]] = None
    image_std: Optional[Tuple[float, float, float]] = None
    # RPN parameters
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 2000
    rpn_post_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_test: int = 2000
    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    rpn_score_thresh: float = 0.05
    # Box parameters
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.5
    bbox_reg_weights: Optional[Tuple[float, float, float, float, float]] = (1, 1, 1, 1, 1)


def get_xy_bounds_text(draw, top_left, text, padding=5):
    x1, y1, x2, y2 = draw.textbbox(
        xy=top_left, text=text, font=FONT, anchor=ANCHOR_TYPE)
    return x1, y1, x2 + padding, y2 + padding


def plot_image(image, output, target):
    image = to_pil_image(image)
    draw = ImageDraw.Draw(image)

    if 'hboxes' in output:
        dt_hboxes = output['hboxes']
        dt_hlabels = output['hlabels']
        dt_hscores = output['hscores']
        hmask = dt_hscores > 0.1
        dt_hboxes = dt_hboxes[hmask].cpu().tolist()
        dt_hlabels = dt_hlabels[hmask].cpu().tolist()
        dt_hscores = dt_hscores[hmask].cpu().tolist()
        for dt_hbox, dt_label, dt_score in zip(dt_hboxes, dt_hlabels, dt_hscores):
            color = MVTecDataset.get_palette(dt_label)
            dt_label = MVTecDataset.idx_to_class(dt_label)
            draw.rectangle(dt_hbox, outline=color, width=5)
            text_to_draw = f'{dt_label} {dt_score:.2f}'

            rectangle = get_xy_bounds_text(draw, dt_hbox[:2], text_to_draw)
            draw.rectangle(rectangle, fill=color)
            draw.text(rectangle[:2], text_to_draw,
                      fill="white", font=FONT, anchor=ANCHOR_TYPE)

    elif 'oboxes' in output:
        dt_oboxes = output['oboxes']
        dt_olabels = output['olabels']
        dt_oscores = output['oscores']
        omask = dt_oscores > 0.1

        dt_oboxes = dt_oboxes[omask].cpu()
        dt_opolys = obb2poly(dt_oboxes).to(int).tolist()
        dt_olabels = dt_olabels[omask].cpu().tolist()
        dt_oscores = dt_oscores[omask].cpu().tolist()

        for dt_opoly, dt_label, dt_score in zip(dt_opolys, dt_olabels, dt_oscores):
            color = MVTecDataset.get_palette(dt_label)
            dt_label = MVTecDataset.idx_to_class(dt_label)
            draw.polygon(dt_opoly, outline=color, width=5)
            text_to_draw = f'{dt_label} {dt_score:.2f}'
            rectangle = get_xy_bounds_text(draw, dt_opoly[:2], text_to_draw)
            draw.rectangle(rectangle, fill=color)
            draw.text(rectangle[:2], text_to_draw,
                      fill="white", font=FONT, anchor=ANCHOR_TYPE)

    gt_boxes = target['bboxes'].cpu().tolist()
    gt_opolys = target['polygons'].cpu().tolist()
    gt_labels = target['labels'].cpu().tolist()

    # TODO include DOTA later

    # gts
    for gt_box, gt_opoly, gt_label in zip(gt_boxes, gt_opolys, gt_labels):
        color = MVTecDataset.get_palette(gt_label)
        gt_label = MVTecDataset.idx_to_class(gt_label)
        draw.rectangle(gt_box, outline=color)
        draw.polygon(gt_opoly, outline=color)
        text_to_draw = f'GT {gt_label}'
        rectangle = get_xy_bounds_text(draw, gt_box[:2], text_to_draw)
        draw.rectangle(rectangle, fill=color)
        draw.text(rectangle[:2], text_to_draw,
                  fill="white", font=FONT, anchor=ANCHOR_TYPE)

    return image, target["image_path"]


class RotatedFasterRCNN(LightningModule):
    def __init__(self, lr: float = 0.001):
        super(RotatedFasterRCNN, self).__init__()
        self.train_config = TrainConfig()
        self.rfrcnn_config = RotatedFasterRCNNConfig(**asdict(MvtecDataConfig()))

        self.model = rotated_fasterrcnn_resnet50_fpn_v2(**asdict(self.train_config), **asdict(self.rfrcnn_config))
        self.lr = lr

        self.outputs = []
        self.targets = []

    def put_outputs(self, outputs):
        # [[cls 1, cls 2 ... cls 13], [(img2)...]]
        for output in outputs:
            per_class_outputs = [[] for _ in range(self.train_config.num_classes - 1)]
            oboxes = output["oboxes"]
            oscores = output["oscores"]
            oboxes_with_scores = torch.cat([oboxes, oscores.unsqueeze(-1)], dim=-1)
            for obox, olabel in zip(oboxes_with_scores, output["olabels"]):
                per_class_outputs[olabel-1].append(obox)
                
            self.outputs.append(per_class_outputs)
        
    def put_targets(self, targets):
        for target in targets:
            target_copy = {}
            target_copy["oboxes"] = target["oboxes"]
            target_copy["labels"] = target["labels"]
            self.targets.append(target_copy)
        
    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self(images, targets)

        loss = sum(loss for loss in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f'train-{k}', v.item())

        self.save_hyperparameters()
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict, outputs = self(images, targets)
        for k, v in loss_dict.items():
            self.log(f'valid-{k}', v.item())

        self.logger.experiment.log({
            "images": [wandb.Image(image, caption=image_path.split('/')[-1])
                       for image, image_path in (plot_image(image, output, target) for image, output, target in zip(images, outputs, targets))]
        })
        
        self.put_outputs(outputs)
        self.put_targets(targets)
        return loss_dict

    def on_validation_epoch_end(self):
        map, _ = eval_rbbox_map(
            self.outputs,
            self.targets, 
            None,
            0.5,
            True
        )
        print(map)
        
        
    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        optimizer = optim.SGD(self.parameters(), lr=self.lr,
                              momentum=0.9, weight_decay=1e-4)
        # return optimizer
        # scheduler = CosineAnnealingWarmUpRestartsDecay(optimizer, T_0=50, T_mult=1, eta_max=0.001,  T_up=10, gamma=0.9)
        scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
        # return [optimizer], [scheduler]
        return [optimizer], [scheduler]
