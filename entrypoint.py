import argparse
import os
import shutil

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from configs import TrainConfig, ModelConfig, Kwargs
from lightning_modules import RotatedFasterRCNN, OrientedRCNN
from datasets.detdemo import DetDemoDataModule, DetDemoDataset
from datasets.mvtec import MVTecDataModule, MVTecDataset
from datasets.dota import DotaDataModule, DotaDataset
from datasets.jmc import JMCDataModule, JMCDataset

def main(args):
    pl.seed_everything(2023)
    torch.set_float32_matmul_precision("medium")
    
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
            f"./datasets/dota_{args.image_size}.pth",
            f"/datasets/split_ss_dota_{args.image_size}",
            train_loader_kwargs,
            test_loader_kwargs
        )
        dataset = DotaDataset

    elif args.dataset == 'mvtec':
        datamodule = MVTecDataModule(
            "./datasets/mvtec_balanced.pth", 
            "/datasets/split_ss_mvtec_balanced", 
            train_loader_kwargs, 
            test_loader_kwargs
        )
        dataset = MVTecDataset
    
    elif args.dataset == 'detdemo':
        datamodule = DetDemoDataModule(
            "./datasets/detdemo.pth",
            "/datasets/detdemo",
            train_loader_kwargs,
            test_loader_kwargs
        )
        dataset = DetDemoDataset

    elif args.dataset == "jmc":
        datamodule = JMCDataModule(
            "./datasets/jmc.pth",
            "/datasets/jmc",
            train_loader_kwargs,
            test_loader_kwargs    
        )
        dataset = JMCDataset
        
    else:
        raise ValueError("Invalid dataset!")
    
    
    datamodule.setup()

    train_config = TrainConfig(
        pretrained=args.pretrained,
        pretrained_backbone=args.pretrained_backbone,
        progress=True,
        num_classes=len(dataset.CLASSES),
        trainable_backbone_layers=args.trainable_backbone_layers,
        learning_rate=args.learning_rate,
        freeze_bn=args.freeze_bn
    )
    
    model_config = ModelConfig(
        min_size = args.image_size,
        max_size = args.image_size,
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
        box_nms_thresh = 0.1,
        box_detections_per_img = 100,
        box_fg_iou_thresh = 0.5,
        box_bg_iou_thresh = 0.5,
        box_batch_size_per_image = 512, # 512
        box_positive_fraction = 0.25,
        bbox_reg_weights = (10, 10, 5, 5, 10),
        
        backbone_type = args.backbone_type
    )
    
    kwargs = Kwargs(
        _skip_flip = args.skip_flip,
        _skip_image_transform = args.skip_image_transform,
        epochs = args.num_epochs,
    )
    steps_per_epoch = len(datamodule.train_dataset) // args.batch_size
    if args.model_type == 'rotated':
        model = RotatedFasterRCNN(
            config={},
            train_config=train_config,
            model_config=model_config,
            kwargs=kwargs,
            dataset=dataset,
            steps_per_epoch=steps_per_epoch
        )
        
    elif args.model_type == 'oriented':
        model = OrientedRCNN(
            config={},
            train_config=train_config,
            model_config=model_config,
            kwargs=kwargs,
            dataset=dataset,
            steps_per_epoch=steps_per_epoch
        )
    
    else:
        raise ValueError("Invalid model type!")
    
    logger = None  
    if args.wandb:
        project_name = f"{args.model_type}-{args.backbone_type}-{args.dataset}-{args.image_size}"
        experiment_name = f"pre:{int(args.pretrained)}_preb:{int(args.pretrained_backbone)}_freezebn:{int(args.freeze_bn)}_skipflip:{int(args.skip_flip)}_skipimgtrf:{int(args.skip_image_transform)}_trainlayers:{args.trainable_backbone_layers}_lr:{args.learning_rate:.6f}"
        logger = WandbLogger(
            project=project_name,
            name=experiment_name,
            log_model=False,
            save_dir="."
        )
        
    
    checkpoint_path = f"./checkpoints/{args.model_type}/{args.backbone_type}/{args.dataset}_{args.image_size}"
    if os.path.exists(checkpoint_path):
        shutil.rmtree(checkpoint_path)
    os.makedirs(checkpoint_path, exist_ok=True)
    
    
    callbacks = [
        ModelCheckpoint(dirpath=checkpoint_path, save_top_k=2, monitor="valid-mAP", mode="max"),
        LearningRateMonitor(logging_interval='step')
    ]

    trainer = pl.Trainer(
        logger=logger, 
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_clip_val, 
        precision=args.precision,
        benchmark=True,
        deterministic=True,
        profiler="advanced",
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    trainer.fit(
        model, 
        datamodule=datamodule,
    )
    
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def validate_args(args):
    if args.model_type == "oriented" and "32" not in args.precision:
        raise ValueError("Oriented R-CNN only supports 32-bit precision!")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotated Faster R-CNN and Oriented R-CNN')
    parser.add_argument('--model_type', type=str, default='rotated', choices=['rotated', 'oriented'],
                        help='Type of model to train (rotated faster r-cnn or oriented r-cnn)')
    parser.add_argument('--backbone_type', type=str, default="mobilenetv3large", choices=["resnet50", "mobilenetv3large", "resnet18", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7"])
    parser.add_argument('--wandb', action='store_true', default=True)
    # Add other necessary arguments
    parser.add_argument('--gradient_clip_val', type=float, default=35.0)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=36)
    parser.add_argument('--dataset', type=str, default='jmc', choices=['mvtec', 'dota', 'detdemo', 'jmc'])
    parser.add_argument('--precision', type=str, default='32', choices=['bf16', 'bf16-mixed', '16', '16-mixed', '32', '32-true', '64', '64-true'])
    
    parser.add_argument('--image_size', type=int, default=256, choices=[256, 512, 800])
    parser.add_argument('--pretrained', type=str2bool, default=False)
    parser.add_argument('--pretrained_backbone', type=str2bool, default=True)
    parser.add_argument('--freeze_bn', type=str2bool, default=False)
    parser.add_argument('--skip_flip', type=str2bool, default=False)
    parser.add_argument('--skip_image_transform', type=str2bool, default=True)
    parser.add_argument('--trainable_backbone_layers', type=int, default=4, choices=[0, 1, 2, 3, 4, 5, 6]) # 5: one batchnorm layer # max 5 for resnet50 and 6 for mobilenetv3large
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    args = parser.parse_args()
    validate_args(args)
    main(args)
