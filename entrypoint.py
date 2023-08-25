import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from oriented_rcnn import OrientedRCNN
from rotated_faster_rcnn import RotatedFasterRCNN
from datasets.mvtec import MVTecDataModule
from datasets.dota import DotaDataModule



def main(args):
    train_loader_kwargs = dict(
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=True, 
        pin_memory=True
    )
    test_loader_kwargs = dict(
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=True
    )

    if args.dataset == 'dota':
        datamodule = DotaDataModule(
            "oc",
            "xyxy",
            "./datasets/dota.pth",
            "/mnt/d/datasets/split_ss_dota",
            train_loader_kwargs,
            test_loader_kwargs
        )
    
    elif args.dataset == 'mvtec':
        datamodule = MVTecDataModule(
            "oc", 
            "xyxy", 
            "./datasets/mvtec.pth", 
            "/mnt/d/datasets/split_ss_mvtec", 
            train_loader_kwargs, 
            test_loader_kwargs
        )
        
    if args.model_type == 'rotated':
        model = RotatedFasterRCNN()
        
    elif args.model_type == 'oriented':
        model = OrientedRCNN()
    else:
        raise ValueError("Invalid model type!")
    
    if args.wandb:
        # arguments made to CometLogger are passed on to the comet_ml.Experiment class
        logger = WandbLogger(
            project=args.project_name,
            name=args.experiment_name,
            log_model=True,
            save_dir="."
        )
        logger.watch(model, log='gradients', log_freq=500, log_graph=True)
        
    else:
        logger = None  
    
    callbacks = [
        ModelCheckpoint(dirpath="./checkpoints", save_top_k=2, monitor="valid-loss", mode="min"),
        LearningRateMonitor(logging_interval='step')
    ]
    
    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_clip_val, 
        precision=args.precision,
        benchmark=False,
        deterministic=False,
        profiler="pytorch",
        # fast_dev_run=True,
        # accelerator="cpu",
        # detect_anomaly=True,
        callbacks=callbacks,
    )
    trainer.fit(
        model, 
        datamodule=datamodule,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotated Faster R-CNN and Oriented R-CNN')
    parser.add_argument('--model_type', type=str, default='oriented', choices=['rotated', 'oriented'],
                        help='Type of model to train (rotated or oriented)')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--project_name', type=str, default='orcnn-implement')
    parser.add_argument('--experiment_name', type=str, default='test upload', help='Leave blank to use default')
    # Add other necessary arguments
    parser.add_argument('--gradient_clip_val', type=float, default=35.0)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=24)
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'dota'])
    parser.add_argument('--precision', type=str, default='32-true', choices=['bf16', 'bf16-mixed', '16', '16-mixed', '32', '32-true', '64', '64-true'])
    
    args = parser.parse_args()
    main(args)
