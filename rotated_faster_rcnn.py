from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Literal

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning import LightningModule
import wandb 

from datasets.dota import DotaDataset
from datasets.mvtec import MVTecDataset
from models.detection.rotated_faster_rcnn import rotated_fasterrcnn_resnet50_fpn
from scheduler import CosineAnnealingWarmUpRestartsDecay, LinearWarmUpMultiStepDecay
from evaluation.evaluator import eval_rbbox_map
from visualize_utils import plot_image

@dataclass
class MvtecDataConfig:
    image_mean: Tuple[float, float, float] = (
        211.35 / 255, 166.559 / 255, 97.271 / 255)
    image_std: Tuple[float, float, float] = (
        43.849 / 255, 40.172 / 255, 30.459 / 255)

@dataclass
class DotaDataConfig:
    image_mean: Tuple[float, float, float] = (
        123.675 / 255, 116.28 / 255, 103.53 / 255)
    image_std: Tuple[float, float, float] = (
        58.395 / 255, 57.12 / 255, 57.375 / 255)

@dataclass
class TrainConfig:
    pretrained: bool = True
    pretrained_backbone: bool = True
    progress: bool = True
    num_classes: int = 13 + 1
    trainable_backbone_layers: Literal[0, 1, 2, 3, 4, 5] = 4
    version: Literal[1, 2] = 2 # TODO: version 1
    learning_rate: float = 0.001

@dataclass
class RotatedFasterRCNNConfig:
    # transform parameters
    min_size: int = 480
    max_size: int = 640
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
    rpn_score_thresh: float = 0.0
    # R-CNN parameters
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25
    bbox_reg_weights: Optional[Tuple[float, float, float, float, float]] = (10, 10, 5, 5, 10)




class RotatedFasterRCNN(LightningModule):
    def __init__(self, config: Optional[dict] = None):
        super(RotatedFasterRCNN, self).__init__()       
        self.train_config = asdict(TrainConfig())
        
        data_config = asdict(MvtecDataConfig())
        self.rfrcnn_config = asdict(RotatedFasterRCNNConfig(**data_config))

        if config:
            config = self._parse_config(config)
            self.train_config.update({k: v for k, v in config.items() if k in self.train_config})
            self.rfrcnn_config.update({k: v for k, v in config.items() if k in self.rfrcnn_config})
            
        self.model = rotated_fasterrcnn_resnet50_fpn(**self.train_config, **self.rfrcnn_config)
        self.lr = self.train_config['learning_rate']

        self.outputs = []
        self.targets = []
    
    def _parse_config(self, config):
        # Avoid in place modification
        config = dict(config.items())
        for k, v in config.items():
            if k in ("pretrained", "pretrained_backbone"):
                config[k] = bool(v)
        return config
                
    def setup(self, stage: Optional[str] = None):
        self.logger.experiment.config.update(self.train_config)
        self.logger.experiment.config.update(self.rfrcnn_config)
        
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
        self.log('train-loss', loss.item())
        # self.save_hyperparameters()
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict, outputs = self(images, targets)
        
        # TODO use evaluation metric mAP to save best model
        loss = sum(loss.detach().item() for loss in loss_dict.values())
        
        for k, v in loss_dict.items():
            self.log(f'valid-{k}', v.item())
        self.log('valid-loss', loss)
        # if batch_idx == 0:
        #     self.logger.experiment.log({
        #         "images": [wandb.Image(pil_image, caption=image_path.split('/')[-1])
        #                 for pil_image, image_path in (plot_image(image, output, target, data=MVTecDataset) for image, output, target in zip(images, outputs, targets))]
        #     })
            
        # self.put_outputs(outputs)
        # self.put_targets(targets)
        return loss_dict

    def on_validation_epoch_end(self):
        # map, _ = eval_rbbox_map(
        #     self.outputs,
        #     self.targets, 
        #     None,
        #     0.5,
        #     True
        # )
        # self.outputs = []
        # self.targets = []
        pass
        
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        # return optimizer
        # scheduler = CosineAnnealingWarmUpRestartsDecay(optimizer, T_0=500, T_mult=1, eta_max=self.lr,  T_up=10, gamma=0.9)
        # scheduler = StepLR(optimizer, step_size=500, gamma=0.5)
        # TODO this is just roughly calculated
        steps_per_epoch = 180
        start, end = steps_per_epoch * 20, steps_per_epoch * 22
        scheduler = LinearWarmUpMultiStepDecay(optimizer, milestones=[start, end], gamma=1/3, warmup_iters=500)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler_config]
