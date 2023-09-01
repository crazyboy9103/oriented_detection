import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import AdvancedProfiler
import wandb

from configs import TrainConfig, ModelConfig, Kwargs
from lightning_modules import RotatedFasterRCNN, OrientedRCNN
from datasets.mvtec import MVTecDataModule, MVTecDataset
from datasets.dota import DotaDataModule, DotaDataset

def main(args):
    wandb.init(project="sweep")
    config = wandb.config
    
    train_loader_kwargs = dict(
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=True, 
        pin_memory=False
    )
    test_loader_kwargs = dict(
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=False
    )

    if args.dataset == 'dota':
        datamodule = DotaDataModule(
            args.data_pth,
            args.data_path,
            train_loader_kwargs,
            test_loader_kwargs
        )
            
        dataset = DotaDataset	
    
    elif args.dataset == 'mvtec':
        datamodule = MVTecDataModule(
            args.data_pth, 
            args.data_path, 
            train_loader_kwargs, 
            test_loader_kwargs
        )

        dataset = MVTecDataset	

    else:
        raise ValueError("Invalid dataset!")
    
    train_config = TrainConfig(
        num_classes=len(dataset.CLASSES),
    )
    
    model_config = ModelConfig(
        min_size = args.image_size,
        max_size = args.image_size,
        image_mean = dataset.IMAGE_MEAN,
        image_std = dataset.IMAGE_STD,
    )

    kwargs = Kwargs()
    
    datamodule.setup()
    steps_per_epoch = len(datamodule.train_dataset) // args.batch_size

    if args.model_type == 'rotated':
        model = RotatedFasterRCNN(
            config=config,
            train_config=train_config,
            model_config=model_config,
            kwargs=kwargs,
            dataset=dataset,
            steps_per_epoch=steps_per_epoch
        )
        
    elif args.model_type == 'oriented':
        model = OrientedRCNN(
            config=config,
            train_config=train_config,
            model_config=model_config,
            kwargs=kwargs,
            dataset=dataset,
            steps_per_epoch=steps_per_epoch
        )
    
    else:
        raise ValueError("Invalid model type!")
    

    logger = WandbLogger(
        project=args.project_name,
        name=args.experiment_name,
        log_model=False,
        save_dir="."
    )
    
    callbacks = [
        LearningRateMonitor(logging_interval='step')
    ]

    
    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_clip_val, 
        precision=args.precision,
        benchmark=True,
        deterministic=True,
        profiler="pytorch",
        callbacks=callbacks,
        detect_anomaly=True
    )
    trainer.fit(
        model, 
        datamodule=datamodule,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rotated', choices=['rotated', 'oriented'])
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'dota'])
    parser.add_argument('--project_name', type=str, default='rfrcnn-sweep')
    parser.add_argument('--experiment_name', type=str, default='test experiment')
    parser.add_argument('--image_size', type=int, default=512, choices=[256, 512, 800])
    parser.add_argument('--data_path', type=str, default='/mnt/d/datasets/split_ss_mvtec')
    parser.add_argument('--data_pth', type=str, default='./datasets/mvtec.pth')
    parser.add_argument('--sweep_name', type=str, default='first_sweep')
    parser.add_argument('--sweep_method', type=str, default='bayes', choices=['random', 'grid', 'bayes'])
    parser.add_argument('--precision', type=str, default='32', choices=['bf16', 'bf16-mixed', '16', '16-mixed', '32', '32-true', '64', '64-true'])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_clip_val', type=float, default=35)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    sweep_config = {
        'method': args.sweep_method,
        'name': args.sweep_name,
        'metric': {
            'goal': 'maximize',
            'name': 'valid-mAP'
        },
        'parameters': {
            'trainable_backbone_layers': {'values': [3, 4, 5]},
            'learning_rate': {'values': [0.00001, 0.0001, 0.001, 0.01]},
                # 'min': 0.0001,
                # 'max': 0.001,
                # 'distribution': 'uniform'

            'pretrained_backbone': {'values': [0, 1]},
            'pretrained': {'values': [0, 1]},
            '_skip_flip': {'values': [0, 1]},
            '_skip_image_transform': {'values': [0, 1]},
            'freeze_bn': {'values': [0, 1]},
        }
    }
    sweep_id=wandb.sweep(sweep_config, project=args.project_name)
    wandb.agent(sweep_id=sweep_id, function=lambda: main(args), count=200)