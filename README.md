# Oriented Object Detection on DOTA/MVTec Screws 

This project is designed for object detection in aerial images and specializes in rotating and oriented bounding boxes. It's built on top of PyTorch and provides Docker support for easy deployment.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup with Docker](#setup-with-docker)
- [Compile csrc](#compile-csrc)
- [Import Library](#import-library)
- [Prepare Dataset](#prepare-dataset)
  - [MVTec Screws](#mvtec-screws)
  - [DOTA v1.5](#dota-v15)
- [Run Training](#run-training)
- [Licenses](#licenses)

## Prerequisites

- Docker (Recommended)
- Python 3.10
- Pip
    - torch 2.0.1
    - torchvision 0.15.2
    - pytorch-lightning 2.0.6
- NVIDIA GPU (Recommended)
    - CUDA 11.7
    - cuDNN 8.5.0

## Setup with Docker (Recommended)

### Build the Docker Image
```
docker build -t <DOCKER_IMAGE_TAG> .
```
Replace `<DOCKER_IMAGE_TAG>` with your desired Docker image tag.

### Run the Docker Container
```
docker run -it -e WANDB_API_KEY="<WANDB_API_KEY>" --gpus all -v <HOST_DATASET_PATH>:/datasets --shm-size=8G --name <CONTAINER_NAME> <DOCKER_IMAGE_TAG>
```
Replace `<WANDB_API_KEY>`, `<HOST_DATASET_PATH>`, and `<CONTAINER_NAME>` with appropriate values.

## Setup with Pip

### Install Dependencies
```
pip install -r requirements.txt
```

## Compile csrc

The `csrc` folder contains custom C++/CUDA source files that need to be compiled. This creates a directory called `mmrotate`, which contains the compiled library.
```
pip install -e .
```

### Import Library

Make sure to import PyTorch before using functions from `mmrotate`. Note that this mmrotate is pointing to the `mmrotate` directory created in the compile step.

```python
import torch
from mmrotate._C import (
    get_compiler_version,
    get_cuda_version,
    has_cuda,
    nms_rotated,
    box_iou_rotated,
    roi_align_rotated_forward,
    roi_align_rotated_backward
)
```
## Prepare dataset
### Dataset folder structure
```
<HOST_DATASET_PATH>/mvtec or dota
    ├─test
    │  ├─annfiles/*.txt 
    │  └─images/*.png
    └─trainval
        ├─annfiles/*.txt 
        └─images/*.png
```

### MVTec Screws
1. Download the dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-screws)
2. Randomly select 10% of the dataset for test set and rest for the trainval set.
3. Place the dataset according to the folder structure above.

### DOTA v1.5
```bibtex
@InProceedings{Xia_2018_CVPR,
    author = {Xia, Gui-Song and Bai, Xiang and Ding, Jian and Zhu, Zhen and Belongie, Serge and Luo, Jiebo and Datcu, Mihai and Pelillo, Marcello and Zhang, Liangpei},
    title = {DOTA: A Large-Scale Dataset for Object Detection in Aerial Images},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
}
```
1. Download the dataset from [here](https://captain-whu.github.io/DOTA/dataset.html) 
2. Note: The test set of DOTA does not have annotations, so we use the validation set as the test set.

## Run Training
Run the training script with the following command:
```
python entrypoint.py \
    --model_type rotated | oriented \ # rotated faster r-cnn | oriented r-cnn
    --wandb \ # use wandb for logging
    --gradient_clip_val 35 \ # gradient clipping
    --batch_size 8 \
    --num_workers 8 \
    --num_epochs 12 \
    --dataset mvtec | dota \ 
    --image_size 256 | 512 | 800 \
    --pretrained True | False \ 
    --pretrained_backbone True | False \ 
    --freeze_bn True | False \ 
    --skip_flip True | False \ 
    --skip_image_transform True | False \ 
    --trainable_backbone_layers 1|2|3|4|5 \ 
    --learning_rate 0.0001
```

| Flag                          | Description                                                  | Options                 |
|-------------------------------|--------------------------------------------------------------|-------------------------|
| `--model_type`                | Choose between rotated or oriented model type                | `rotated`, `oriented`   |
| `--wandb`                     | Use Weights and Biases for logging                           |                         |
| `--batch_size`                | Set the batch size for training                              | Numerical values > 0    |
| `--num_epochs`                | Set the number of training epochs                            | Numerical values > 0    |
| `--dataset`                   | Choose between mvtec or dota dataset                         | `mvtec`, `dota`         |  
| `--image_size`                | Set the image size for training                              | `256`, `512`, `800`     |
| `--pretrained`                | Use COCO pretrained backbone + FPN                           | `True`, `False`         |
| `--pretrained_backbone`       | Use ImageNet pretrained backbone  (overridden by pretrained) | `True`, `False`         |
| `--freeze_bn`                 | Freeze batch normalization layers                            | `True`, `False`         |
| `--skip_flip`                 | Skip random horizontal flipping                              | `True`, `False`         |
| `--skip_image_transform`      | Skip random image transformation                             | `True`, `False`         |
| `--trainable_backbone_layers` | Set the number of trainable backbone layers                  | `1`, `2`, `3`, `4`, `5` |
| `--learning_rate`             | Set the learning rate for training                           | Numerical values > 0    |

## Licenses
This project integrates code from the following two different projects:

1. torchvision is licensed under the BSD 3-Clause License. [Link to full license](https://github.com/pytorch/vision/blob/main/LICENSE)
2. mmrotate is licensed under the Apache License 2.0. [Link to full license](https://github.com/open-mmlab/mmrotate/blob/main/LICENSE)