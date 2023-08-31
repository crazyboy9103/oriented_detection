from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
import wandb 

from configs import TrainConfig, ModelConfig, Kwargs
from datasets.dota import DotaDataset
from datasets.mvtec import MVTecDataset
from models.detection.builder import faster_rcnn_builder, oriented_rcnn_builder
from scheduler import CosineAnnealingWarmUpRestartsDecay, LinearWarmUpMultiStepDecay
from evaluation.evaluator import eval_rbbox_map
from visualize_utils import plot_image

class ModelWrapper(LightningModule):
    def __init__(
        self, 
        config: Optional[dict] = None,
        train_config: Optional[dataclass] = None, 
        model_config: Optional[dataclass] = None,
        kwargs: Optional[dataclass] = None,
        dataset: DotaDataset|MVTecDataset = MVTecDataset,
        steps_per_epoch: int = None
    ):
        super(ModelWrapper, self).__init__()
        
        if train_config is None:
            train_config = TrainConfig()
        
        if model_config is None:
            model_config = ModelConfig(
                image_mean=dataset.IMAGE_MEAN,
                image_std=dataset.IMAGE_STD,
            )
        
        if kwargs is None:
            kwargs = Kwargs()
         
        self.train_config = asdict(train_config)
        self.model_config = asdict(model_config)
        self.kwargs = asdict(kwargs)
        self.dataset = dataset
        
        if config:
            self.config = self._parse_config(config)
            
            self.train_config.update({k: v for k, v in self.config.items() if k in self.train_config})
            self.model_config.update({k: v for k, v in self.config.items() if k in self.model_config})
            self.kwargs.update({k: v for k, v in self.config.items() if k in self.kwargs})
        
        self.lr = self.train_config['learning_rate']
        self.steps_per_epoch = steps_per_epoch
        
        self.outputs = []
        self.targets = []
        
    
    def _parse_config(self, config):
        # Avoid in place modification
        # As wandb config does not accept boolean values, they must be manually converted
        config = dict(config.items())
        for k, v in config.items():
            if k in ("pretrained", "pretrained_backbone", "_skip_flip", "_skip_image_transform"):
                config[k] = bool(v)
        return config
                
    def setup(self, stage: Optional[str] = None):
        self.logger.experiment.config.update(self.train_config)
        self.logger.experiment.config.update(self.model_config)
        self.logger.experiment.config.update(self.kwargs)
        
    def put_outputs(self, outputs):
        # [[cls 1, cls 2 ... cls 13], [(img2)...]]
        for output in outputs:
            per_class_outputs = [[] for _ in range(self.train_config['num_classes'] - 1)]
            oboxes = output["oboxes"].detach().cpu()
            oscores = output["oscores"].detach().cpu()
            olabels = output["olabels"].detach().cpu()
            oboxes_with_scores = torch.cat([oboxes, oscores.unsqueeze(-1)], dim=-1)
            
            for obox, olabel in zip(oboxes_with_scores, olabels):
                per_class_outputs[olabel-1].append(obox)
                
            self.outputs.append(per_class_outputs)
        
    def put_targets(self, targets):
        for target in targets:
            target_copy = {}
            target_copy["oboxes"] = target["oboxes"].detach().cpu()
            target_copy["labels"] = target["labels"].detach().cpu()
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
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        loss_dict, outputs = self(images, targets)
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        print(f"Elapsed time {elapsed_time:.6f}ms")
        print(f"Im/s: {len(images)/(elapsed_time / 1000):.6f}")
        loss = sum(loss.item() for loss in loss_dict.values())
        
        for k, v in loss_dict.items():
            self.log(f'valid-{k}', v.item())
            
        self.log('valid-loss', loss)
        if not hasattr(self, 'config') and batch_idx == 0:
            self.logger.experiment.log({
                "images": [wandb.Image(pil_image, caption=image_path.split('/')[-1])
                        for pil_image, image_path in (plot_image(image, output, target, self.dataset, 0.5, 0.5) for image, output, target in zip(images, outputs, targets))]
            })
            
        self.put_outputs(outputs)
        self.put_targets(targets)
        return loss_dict

    def on_validation_epoch_end(self):
        map, _ = eval_rbbox_map(
            self.outputs,
            self.targets, 
            0.5,
            True
        )
        self.log('valid-mAP', map)
        self.outputs = []
        self.targets = []
        
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        start, end = self.steps_per_epoch * 8, self.steps_per_epoch * 10
        scheduler = LinearWarmUpMultiStepDecay(optimizer, milestones=[start, end], gamma=1/3, warmup_iters=500)
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler_config]


class RotatedFasterRCNN(ModelWrapper):
    def __init__(
        self, 
        config: Optional[dict] = None,
        train_config: Optional[dataclass] = None, 
        model_config: Optional[dataclass] = None,
        kwargs: Optional[dataclass] = None,
        dataset: DotaDataset|MVTecDataset = MVTecDataset,
        steps_per_epoch: int = None
    ):
        super(RotatedFasterRCNN, self).__init__(config, train_config, model_config, kwargs, dataset, steps_per_epoch)
        self.model = faster_rcnn_builder(**self.train_config, **self.model_config, kwargs=self.kwargs)

class OrientedRCNN(ModelWrapper):
    def __init__(
        self, 
        config: Optional[dict] = None,
        train_config: Optional[dataclass] = None, 
        model_config: Optional[dataclass] = None,
        kwargs: Optional[dataclass] = None,
        dataset: DotaDataset|MVTecDataset = MVTecDataset,
        steps_per_epoch: int = None
    ):
        super(OrientedRCNN, self).__init__(config, train_config, model_config, kwargs, dataset, steps_per_epoch)
        self.model = oriented_rcnn_builder(**self.train_config, **self.model_config, kwargs=self.kwargs)
    
