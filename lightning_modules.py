from dataclasses import dataclass, asdict
from typing import Optional
import gc

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
import wandb 

from configs import TrainConfig, ModelConfig, Kwargs
from datasets.dota import DotaDataset
from datasets.mvtec import MVTecDataset
from models.detection.builder import faster_rcnn_builder, oriented_rcnn_builder
from scheduler import LinearWarmUpMultiStepDecay
from evaluation.neurocle_evaluator import DetectionEvaluator, NeurocleDetectionEvaluator
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
        self.total_epochs = self.kwargs['epochs']
        
        self.steps_per_epoch = steps_per_epoch
        
        self.detection_evaluator = DetectionEvaluator(
            iou_threshold=0.5,
            rotated=True, 
            num_classes=self.train_config['num_classes']-1
        )
        
        self.neurocle_detection_evaluator = NeurocleDetectionEvaluator(0.5, 0.5)
    
    def _parse_config(self, config):
        # Avoid in place modification
        # As wandb config does not accept boolean values, they must be manually converted
        config = dict(config.items())
        for k, v in config.items():
            if k == "trainable_backbone_layers":
                config[k] = int(v)
            
            elif k == "learning_rate":
                config[k] = float(v)
            
            elif k in ("pretrained", "pretrained_backbone", "_skip_flip", "_skip_image_transform", "freeze_bn"):
                config[k] = config[k] == "True"
                
        return config
                
    def setup(self, stage: Optional[str] = None):
        self.logger.experiment.config.update(self.train_config)
        self.logger.experiment.config.update(self.model_config)
        self.logger.experiment.config.update(self.kwargs)
        
    def forward(self, images, targets=None):
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_dict = self(images, targets)
        
        # self.current_epoch 
        loss = sum(loss for loss in loss_dict.values())

        for k, v in loss_dict.items():
            self.log(f'train-{k}', v.item())
        self.log('train-loss', loss.item())
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
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
        self.log('FPS', len(images)/(elapsed_time / 1000))
        loss = sum(loss.item() for loss in loss_dict.values())
        for k, v in loss_dict.items():
            self.log(f'valid-{k}', v.item())
            
        self.log('valid-loss', loss)
        # loss_dict.update({'valid-loss': loss})
        # skip image logging for sweeps
        if not hasattr(self, 'config') and batch_idx == 0:
            self.logger.experiment.log({
                "images": [
                    wandb.Image(pil_image, caption=image_path.split('/')[-1])
                    for pil_image, image_path in (
                        plot_image(image, output, target, self.dataset, 0.5) for image, output, target in zip(images, outputs, targets)
                    )
                ]
            })
            
        self.detection_evaluator.accumulate(targets, outputs)
        self.neurocle_detection_evaluator.update_state(targets, outputs)
        
        # return loss_dict

    def on_validation_epoch_end(self):
        aggregate_metrics, detailed_metrics= self.detection_evaluator.compute_metrics()
        print("detailed_metrics", detailed_metrics)
        print("aggregate_metrics", aggregate_metrics)
        
        for key, value in aggregate_metrics.items():
            self.log(f"valid-{key}", value)
        
        self.detection_evaluator.reset()
        
        neurocle_result = self.neurocle_detection_evaluator.result()
        for key, value in neurocle_result.items():
            self.log(f"valid-{key}", value)
        print("neurocle_result", neurocle_result)
        self.neurocle_detection_evaluator.reset_states()
        
        torch.cuda.empty_cache()
        gc.collect()
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # following milestones, warmup_iters are arbitrarily chosen
        first, second = self.steps_per_epoch * int(self.total_epochs * 4/6), self.steps_per_epoch * int(self.total_epochs * 5/6)
        warmup_iters = self.steps_per_epoch * int(self.total_epochs * 1/6)
        scheduler = LinearWarmUpMultiStepDecay(optimizer, milestones=[first, second], gamma=1/3, warmup_iters=warmup_iters)
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
    
