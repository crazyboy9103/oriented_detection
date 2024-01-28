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
    - see requirements.txt
- NVIDIA GPU
    - CUDA 11.8
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

## Setup with Pip

### Install Dependencies
```
pip install -r requirements.txt
```

## Compile csrc

The `csrc` folder contains custom C++/CUDA source files that need to be compiled. This creates a directory called `detectron2`, which contains the compiled library.
```
pip install -e .
```

### Import Library

Make sure to import PyTorch before using functions from `detectron2`. Note that this detectron2 is pointing to the `detectron2` directory created in the compile step.

```python
import torch
from detectron2._C import (
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
See entrypoint.py for arguments

## Licenses
This project adaptes code from the following two different projects:

1. torchvision is licensed under the BSD 3-Clause License. [Link to full license](https://github.com/pytorch/vision/blob/main/LICENSE)
2. detectron2 is licensed under the Apache License 2.0. [Link to full license](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)