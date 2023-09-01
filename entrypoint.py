import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import AdvancedProfiler

from configs import TrainConfig, ModelConfig, Kwargs
from lightning_modules import RotatedFasterRCNN, OrientedRCNN
from datasets.mvtec import MVTecDataModule, MVTecDataset
from datasets.dota import DotaDataModule, DotaDataset

def main(args):
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
            "./datasets/dota_256.pth",
            "/mnt/d/datasets/split_ss_dota_256",
            train_loader_kwargs,
            test_loader_kwargs
        )
        dataset = DotaDataset

    elif args.dataset == 'mvtec':
        datamodule = MVTecDataModule(
            "./datasets/mvtec.pth", 
            "/mnt/d/datasets/split_ss_mvtec", 
            train_loader_kwargs, 
            test_loader_kwargs
        )
        dataset = MVTecDataset
    
    else:
        raise ValueError("Invalid dataset!")
    
    train_config = TrainConfig(
        pretrained=True,
        pretrained_backbone=True,
        progress=True,
        num_classes=len(dataset.CLASSES),
        trainable_backbone_layers=4,
        version=2,
        learning_rate=0.0001,
        freeze_bn=False
    )
    
    model_config = ModelConfig(
        min_size = 256,
        max_size = 256,
        image_mean = dataset.IMAGE_MEAN,
        image_std = dataset.IMAGE_STD,
        rpn_pre_nms_top_n_train = 2000,
        rpn_pre_nms_top_n_test = 2000,
        rpn_post_nms_top_n_train = 2000,
        rpn_post_nms_top_n_test = 2000,
        rpn_nms_thresh = 0.7,
        rpn_fg_iou_thresh = 0.7,
        rpn_bg_iou_thresh = 0.3,
        rpn_batch_size_per_image = 256,
        rpn_positive_fraction = 0.5,
        rpn_score_thresh = 0,
        
        box_score_thresh = 0.05,
        box_nms_thresh = 0.5,
        box_detections_per_img = 100,
        box_fg_iou_thresh = 0.5,
        box_bg_iou_thresh = 0.5,
        box_batch_size_per_image = 512, # 512
        box_positive_fraction = 0.25,
        bbox_reg_weights = (10, 10, 5, 5, 10)
    )

    kwargs = Kwargs(
        _skip_flip = False,
        _skip_image_transform = True,
        epochs = args.num_epochs,
    )
    datamodule.setup()
    steps_per_epoch = len(datamodule.train_dataset) // args.batch_size
    if args.model_type == 'rotated':
        model = RotatedFasterRCNN(
            train_config=train_config,
            model_config=model_config,
            kwargs=kwargs,
            dataset=dataset,
            steps_per_epoch=steps_per_epoch
        )
        
    elif args.model_type == 'oriented':
        model = OrientedRCNN(
            train_config=train_config,
            model_config=model_config,
            kwargs=kwargs,
            dataset=dataset,
            steps_per_epoch=steps_per_epoch
        )
    
    else:
        raise ValueError("Invalid model type!")
    
    if args.wandb:
        # arguments made to CometLogger are passed on to the comet_ml.Experiment class
        logger = WandbLogger(
            project=args.project_name,
            name=args.experiment_name,
            log_model=False,
            save_dir="."
        )
        logger.watch(model, log='gradients', log_freq=500, log_graph=True)
        
    else:
        logger = None  
    
    callbacks = [
        ModelCheckpoint(dirpath="./checkpoints", save_top_k=2, monitor="valid-loss", mode="min"),
        LearningRateMonitor(logging_interval='step')
    ]

    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    
    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_clip_val, 
        precision=args.precision,
        benchmark=True,
        deterministic=True,
        profiler=profiler,
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
                        help='Type of model to train (rotated faster r-cnn or oriented r-cnn)')
    parser.add_argument('--wandb', action='store_true', default=True)
    parser.add_argument('--project_name', type=str, default='orcnn-implement')
    parser.add_argument('--experiment_name', type=str, default='mvtec256_16bit', help='Leave blank to use default')
    # Add other necessary arguments
    parser.add_argument('--gradient_clip_val', type=float, default=35.0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=12)
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'dota'])
    parser.add_argument('--precision', type=str, default='32', choices=['bf16', 'bf16-mixed', '16', '16-mixed', '32', '32-true', '64', '64-true'])
    args = parser.parse_args()
    main(args)
