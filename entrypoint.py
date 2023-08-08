import argparse
import pytorch_lightning as pl
from rotated_faster_rcnn import RotatedFasterRCNN
from oriented_rcnn import OrientedRCNN

def main(args):
    if args.model_type == 'rotated':
        model = RotatedFasterRCNN()
    elif args.model_type == 'oriented':
        model = OrientedRCNN()
    else:
        raise ValueError("Invalid model type!")

    trainer = pl.Trainer()
    trainer.fit(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rotated Faster R-CNN and Oriented R-CNN')
    parser.add_argument('--model_type', type=str, default='rotated', choices=['rotated', 'oriented'],
                        help='Type of model to train (rotated or oriented)')
    # Add other necessary arguments

    args = parser.parse_args()
    main(args)
