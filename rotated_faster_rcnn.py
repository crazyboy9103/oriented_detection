from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch.optim as optim
from pytorch_lightning import LightningModule
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw

from models.detection.rotated_faster_rcnn import rotated_fasterrcnn_resnet50_fpn_v2
from scheduler import CosineAnnealingWarmUpRestartsDecay
from ops.boxes import obb2poly
from datasets.mvtec import MVTecDataset
from datasets.dota import DotaDataset
# TODO add more args
@dataclass
class MvtecDataConfig:
    image_mean: Tuple[float, float, float] = (211.35, 166.559, 97.271)
    image_std: Tuple[float, float, float] = (43.849, 40.172, 30.459)
    
@dataclass
class TrainConfig:
    pretrained: bool = False
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
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_test: int = 1000
    rpn_nms_thresh: float = 0.5
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
    
    
def plot_image(image, output, target):
    image = to_pil_image(image)
    draw = ImageDraw.Draw(image)
    
    dt_boxes = output['bboxes']
    dt_oboxes = output['oboxes']
    dt_labels = output['labels']
    dt_scores = output['scores']
    mask = dt_scores > 0.2
    
    dt_boxes = dt_boxes[mask].cpu().tolist()
    dt_oboxes = dt_oboxes[mask].cpu()
    dt_opolys = obb2poly(dt_oboxes).tolist()
    dt_labels = dt_labels[mask].cpu().tolist()
    dt_scores = dt_scores[mask].cpu().tolist()
    
    gt_boxes = target['bboxes'].cpu().tolist()
    gt_opolys = target['polygons'].cpu().tolist()
    gt_labels = target['labels'].cpu().tolist()
    
    
    # TODO include DOTA later
    # detections
    for dt_box, dt_opoly, dt_label, dt_score in zip(dt_boxes, dt_opolys, dt_labels, dt_scores):
        color = MVTecDataset.get_palette(dt_label)
        dt_label = MVTecDataset.idx_to_class(dt_label)
        draw.rectangle(dt_box, outline=color, width=3)
        draw.polygon(dt_opoly, outline=color, width=3)
        draw.text(dt_box[:2], f'{dt_label} {dt_score:.2f}%', fill=color)
    
    # gts
    for gt_box, gt_opoly, gt_label in zip(gt_boxes, gt_opolys, gt_labels):
        color = MVTecDataset.get_palette(gt_label)
        gt_label = MVTecDataset.idx_to_class(gt_label)
        draw.rectangle(gt_box, outline=color)
        draw.polygon(gt_opoly, outline=color)
        draw.text(gt_box[:2], f'GT {gt_label}', fill=color)

    return image, target["image_path"]

class RotatedFasterRCNN(LightningModule):
    def __init__(self, lr: float = 0.001):
        super(RotatedFasterRCNN, self).__init__()
        self.model = rotated_fasterrcnn_resnet50_fpn_v2(**asdict(TrainConfig()), **asdict(RotatedFasterRCNNConfig(**asdict(MvtecDataConfig()))))
        self.lr = lr
        self.save_hyperparameters()
        
        self.outputs = []
        
    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self(images, targets)
        
        loss = sum(loss for loss in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f'train-{k}', v.item())
            
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict, outputs = self(images, targets)
        for k, v in loss_dict.items():
            self.log(f'valid-{k}', v.item())
        self.outputs.append(outputs)
        
        for image, image_path in (plot_image(image, output, target) for image, output, target in zip(images, outputs, targets)):
            self.logger.experiment.log_image(image, name=image_path)
        return loss_dict    
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer
        # scheduler = CosineAnnealingWarmUpRestartsDecay(optimizer, T_0=50, T_mult=1, eta_max=0.1,  T_up=10, gamma=0.5)
        # return [optimizer], [scheduler]


