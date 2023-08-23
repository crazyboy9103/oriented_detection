import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import argparse

# from oriented_rcnn import OrientedRCNN
from rotated_faster_rcnn import RotatedFasterRCNN
from datasets.mvtec import MVTecDataModule
from datasets.dota import DotaDataModule

def main(args):
    wandb.init(project="sweep")
    config = wandb.config
    
    train_loader_kwargs = dict(
        batch_size=config.batch_size, 
        num_workers=args.num_workers, 
        shuffle=True, 
        pin_memory=True
    )
    test_loader_kwargs = dict(
        batch_size=config.batch_size, 
        num_workers=args.num_workers, 
        shuffle=False, 
        pin_memory=True
    )

    if args.dataset == 'dota':
        raise NotImplementedError
    
    elif args.dataset == 'mvtec':
        datamodule = MVTecDataModule(
            "oc", 
            "xyxy", 
            "/workspace/datasets/mvtec.pth", 
            args.data_path, 
            train_loader_kwargs, 
            test_loader_kwargs
        )
#     wandb.require(experiment="service")

    if args.model_type == 'rotated':
        model = RotatedFasterRCNN(config)
    elif args.model_type == 'oriented':
        # model = OrientedRCNN()
        raise NotImplementedError
    else:
        raise ValueError("Invalid model type!")
    
    
    # arguments made to CometLogger are passed on to the comet_ml.Experiment class
    logger = WandbLogger(
        project=args.project_name,
        name=args.experiment_name,
        log_model=True,
        save_dir="."
    )
    # logger.watch(model, log='gradients', log_freq=500, log_graph=True)
    
    
    callbacks = [
        # ModelCheckpoint(dirpath=args.checkpoint_path, save_top_k=2, monitor="valid-loss", mode="min"),
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rotated', choices=['rotated', 'oriented'])
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'dota'])
    parser.add_argument('--project_name', type=str, default='rfrcnn-sweep')
    parser.add_argument('--experiment_name', type=str, default='test experiment')
    parser.add_argument('--data_path', type=str, default='/default/datasets/mvtec-rotated-screws')
    parser.add_argument('--sweep_name', type=str, default='first_sweep')
    parser.add_argument('--sweep_method', type=str, default='random', choices=['random', 'grid', 'bayes'])
    parser.add_argument('--precision', type=str, default='32-true', choices=['bf16', 'bf16-mixed', '16', '16-mixed', '32', '32-true', '64', '64-true'])
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_clip_val', type=float, default=35)
    parser.add_argument('--num_epochs', type=int, default=24)
    args = parser.parse_args()
    
    sweep_config = {
        'method': args.sweep_method,
        'name': args.sweep_name,
        'metric': {
            'goal': 'minimize',
            'name': 'valid-loss'
        },
        'parameters': {
            'batch_size': {'values': [2, 4]},
            'trainable_backbone_layers': {'values': [0, 1, 2, 3, 4, 5]},
            'learning_rate': {
                'min': 0.00001,
                'max': 0.1,
                'distribution': 'uniform'
            },
            'pretrained_backbone': {'values': [0, 1]},
            'pretrained': {'values': [0, 1]},
            '_skip_flip': {'values': [0, 1]},
            '_skip_image_transform': {'values': [0, 1]},
            'box_positive_fraction': {'values': [0.25, 0.5]},
            'box_detections_per_img': {'values': [100, 200]},
        }
    }
    sweep_id=wandb.sweep(sweep_config, project=args.project_name)
    wandb.agent(sweep_id=sweep_id, function=lambda: main(args), count=5)
    