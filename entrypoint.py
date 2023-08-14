import argparse

import pytorch_lightning as pl
# from oriented_rcnn import OrientedRCNN
from rotated_faster_rcnn import RotatedFasterRCNN
from pytorch_lightning.loggers import WandbLogger, CometLogger

from datasets.mvtec import MVTecDataModule
from datasets.dota import DotaDataModule




def main(args):
    train_loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    if args.dataset == 'dota':
        raise NotImplementedError
    
    elif args.dataset == 'mvtec':
        datamodule = MVTecDataModule(
            "oc", 
            "xyxy", 
            "/workspace/datasets/mvtec.pth", 
            "/datasets/mvtec_screws", 
            train_loader_kwargs, 
            test_loader_kwargs
        )
        
    if args.model_type == 'rotated':
        model = RotatedFasterRCNN(lr=args.lr)
    elif args.model_type == 'oriented':
        # model = OrientedRCNN()
        raise NotImplementedError
    else:
        raise ValueError("Invalid model type!")
    
    if args.comet:
        # arguments made to CometLogger are passed on to the comet_ml.Experiment class
        comet_logger = CometLogger(
            api_key=args.comet_api_key,
            workspace=args.comet_workspace,  # Optional
            save_dir=".",  # Optional
            project_name=args.comet_project_name,  # Optional
            experiment_name="lightning_logs" if not args.experiment_name.replace(" ", "") else args.experiment_name,  # Optional
        )
    else:
        comet_logger = None  
        
    trainer = pl.Trainer(
        logger=comet_logger, 
        max_epochs=args.num_epochs,
        gradient_clip_val=args.gradient_clip_val, 
        precision=args.precision,
        benchmark=False,
        deterministic=False,
        profiler="advanced"
    )
    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotated Faster R-CNN and Oriented R-CNN')
    parser.add_argument('--model_type', type=str, default='rotated', choices=['rotated', 'oriented'],
                        help='Type of model to train (rotated or oriented)')
    parser.add_argument('--comet', action='store_true', default=True)
    parser.add_argument('--comet_api_key', type=str, default='zE32bzu4ngD8Clz4WKG7r9u9B')
    parser.add_argument('--comet_workspace', type=str, default='crazyboy9103')
    parser.add_argument('--comet_project_name', type=str, default='rfrcnn-implement')
    parser.add_argument('--experiment_name', type=str, default='test upload', help='Leave blank to use default')
    # Add other necessary arguments
    parser.add_argument('--gradient_clip_val', type=float, default=35.0)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'dota'])
    parser.add_argument('--precision', type=str, default='32-true', choices=['bf16', 'bf16-mixed', '16', '16-mixed', '32', '32-true', '64', '64-true'])
    
    args = parser.parse_args()
    main(args)
