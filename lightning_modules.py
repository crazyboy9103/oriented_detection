from dataclasses import dataclass, asdict
from typing import Optional, Type
import gc

import torch
import torch.optim as optim
from pytorch_lightning import LightningModule
import wandb 

from configs import TrainConfig, ModelConfig, Kwargs
from datasets.base import BaseDataset
from models.detection.builder import rotated_faster_rcnn_builder
from scheduler import LinearWarmUpMultiStepDecay
from evaluation.neurocle_evaluator import DetectionEvaluator, NeurocleDetectionEvaluator
from visualize_utils import plot_image
        
        
class ModelWrapper(LightningModule):
    def __init__(
        self, 
        train_config: Optional[dataclass], 
        model_config: Optional[dataclass],
        kwargs: Optional[dataclass],
        dataset: Type[BaseDataset],
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
        
        self.lr = self.train_config['learning_rate']
        
        self.detection_evaluator = DetectionEvaluator(
            iou_threshold=0.5,
            rotated=True, 
            num_classes=self.train_config['num_classes']-1
        )
        
        # self.neurocle_detection_evaluator = NeurocleDetectionEvaluator(0.5, 0.5)
    
    def setup(self, stage: Optional[str] = None):
        if hasattr(self.logger.experiment, "config"):
            self.logger.experiment.config.update(self.train_config)
            self.logger.experiment.config.update(self.model_config)
            self.logger.experiment.config.update(self.kwargs)
            
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
        print("loss", loss.item())
        return loss

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
        gc.collect()
        return super().on_train_epoch_end()
        
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        
        loss_dict, outputs = self(images, targets)
        for k, v in loss_dict.items():
            self.log(f'valid-{k}', v.item())
        
        loss = sum(loss.item() for loss in loss_dict.values())
        self.log('valid-loss', loss)
        # skip image logging for sweeps
        if not hasattr(self, 'config') and batch_idx == self.trainer.num_val_batches[0]-3:
            size = self.model_config.get("min_size") # * 2 # upsample for better visualization
            resize_image = (size, size)
            self.logger.experiment.log({
                "images": [
                    wandb.Image(pil_image, caption=image_path.split('/')[-1])
                    for pil_image, image_path in (
                        plot_image(image, output, target, self.dataset, 0.5, resize=None) for image, output, target in zip(images, outputs, targets)
                    )
                ]
            })
            
        self.detection_evaluator.accumulate(targets, outputs)
        # self.neurocle_detection_evaluator.update_state(targets, outputs)
        
    def on_validation_epoch_end(self):
        aggregate_metrics, detailed_metrics= self.detection_evaluator.compute_metrics()
        print("detailed_metrics", detailed_metrics)
        print("aggregate_metrics", aggregate_metrics)
        
        for key, value in aggregate_metrics.items():
            self.log(f"valid-{key}", value)
        
        self.detection_evaluator.reset()
        
        torch.cuda.empty_cache()
        gc.collect()
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        # following milestones, warmup_iters are arbitrarily chosen
        first, second = int(steps_per_epoch * self.trainer.max_epochs * 4/6), int(steps_per_epoch * self.trainer.max_epochs * 5/6)
        warmup_iters = steps_per_epoch // 10
        scheduler = LinearWarmUpMultiStepDecay(
            optimizer, 
            milestones=[first, second], 
            gamma=0.1, 
            warmup_start_lr=0,
            warmup_iters=warmup_iters,
        )
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }
        return [optimizer], [scheduler_config]


class RotatedFasterRCNN(ModelWrapper):
    def __init__(
        self, 
        train_config, 
        model_config,
        kwargs,
        dataset,
    ):
        super(RotatedFasterRCNN, self).__init__(train_config, model_config, kwargs, dataset)
        self.model = rotated_faster_rcnn_builder(**self.train_config, **self.model_config, kwargs=self.kwargs)