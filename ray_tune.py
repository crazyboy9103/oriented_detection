from functools import partial
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import ray
from ray import tune
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, CategoricalHyperparameter

from configs import TrainConfig, ModelConfig, Kwargs
from lightning_modules import RotatedFasterRCNN, OrientedRCNN
from datasets.mvtec import MVTecDataModule, MVTecDataset
from datasets.dota import DotaDataModule, DotaDataset

def train_tune(config, args):
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
        
    wandb_project_name = f"{args.model_type}-{args.dataset}-{args.image_size}-tune"
    experiment_name = f"#-{config['trainable_backbone_layers']}_lr-{config['learning_rate']:.6f}_preb-{config['pretrained_backbone']}_pre-{config['pretrained']}_sf-{config['_skip_flip']}_si-{config['_skip_image_transform']}_fbn-{config['freeze_bn']}"
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
    ]
    logger = WandbLogger(
        project=wandb_project_name,
        name=experiment_name,
        log_model=False,
        save_dir="."
    )
    logger.watch(model, log='gradients', log_freq=20, log_graph=True)

    trainer = pl.Trainer(
        logger=logger,
        log_every_n_steps=steps_per_epoch,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_clip_val, 
        precision=args.precision,
        benchmark=True,
        profiler="pytorch",
        accelerator="gpu",
        callbacks=callbacks,
    )
    trainer.fit(
        model, 
        datamodule=datamodule,
    )
    
def main(args):
    # config space
    config_space = ConfigurationSpace(name="config-space", seed=2023)
    config_space.add_hyperparameters([
        CategoricalHyperparameter("trainable_backbone_layers", choices=[1, 2, 3, 4, 5]),
        UniformFloatHyperparameter("learning_rate", lower=0.00001, upper=0.01),
        CategoricalHyperparameter("pretrained_backbone", choices=[True, False]),
        CategoricalHyperparameter("pretrained", choices=[True, False]),
        CategoricalHyperparameter("_skip_flip", choices=[True, False]),
        CategoricalHyperparameter("_skip_image_transform", choices=[True, False]),
        CategoricalHyperparameter("freeze_bn", choices=[True, False]),
    ])
    
    # BOHB config
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=1200,
        reduction_factor=4,
        metric="valid-mAP",
        mode="max"
    )
    bohb_search = TuneBOHB(
        config_space,
        max_concurrent=1,
        metric="valid-mAP",
        mode="max"
    )

    train_tune_with_args = partial(train_tune, args=args)
    wandb_project_name = f"{args.model_type}-{args.dataset}-{args.image_size}-tune"
    
    # Ray Tune config
    analysis = tune.run(
        train_tune_with_args,
        resources_per_trial={"gpu": 1, "cpu": args.num_workers * 2},
        num_samples=50,
        scheduler=bohb_hyperband,
        search_alg=bohb_search,
        callbacks=[
            WandbLoggerCallback(
                project = wandb_project_name,
                group= "tuning",
                # api_key: Optional[str] = None,
                log_config = True,
                save_checkpoints = True,
            )
        ],
    )
    
    print(analysis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rotated', choices=['rotated', 'oriented'])
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'dota'])
    parser.add_argument('--image_size', type=int, default=512, choices=[256, 512, 800])
    parser.add_argument('--data_path', type=str, default='/mnt/d/datasets/split_ss_mvtec')
    parser.add_argument('--data_pth', type=str, default='./datasets/mvtec.pth')
    parser.add_argument('--precision', type=str, default='32-true', choices=['bf16', 'bf16-mixed', '16', '16-mixed', '32', '32-true', '64', '64-true'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gradient_clip_val', type=float, default=35)
    parser.add_argument('--num_epochs', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    ray.init(num_gpus=1)
    main(args)