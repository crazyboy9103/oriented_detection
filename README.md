# Dockerfile
```shell
docker build -t ${DOCKER IMAGE TAG} .
docker run -it -e WANDB_API_KEY="${WANDB API KEY}" --gpus all -v ${HOST DATASET PATH}:/datasets --shm-size=8G --name ${CONTAINER NAME} ${DOCKER IMAGE TAG}
```
# Compile csrc
```shell
pip install -e .
```

```python
import torch # torch must be imported before mmrotate
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
Dataset folder structure:
```
${HOST DATASET PATH}/mvtec or dota
                        ├─test
                        │  ├─annfiles/*.txt 
                        │  └─images/*.png
                        └─trainval
                            ├─annfiles/*.txt 
                            └─images/*.png
```
### MVTec Screws
1. Download dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-screws)
2. Randomly select 10% of the dataset as test set and rest as trainval set
3. Place the dataset in the folder structure shown above

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
* We split images into patches of desired resolutions for DOTA dataset. Refer to [this repo](https://github.com/jbwang1997/BboxToolkit/tree/master).
    * It is important to note that the test set of DOTA does not have annotations. Therefore, we use the validation set as test set.

1. Download dataset from [here](https://captain-whu.github.io/DOTA/dataset.html) 
2.  
## Run training
```python
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

## Licenses
This project integrates code from two different projects:

1. torchvision is licensed under the BSD 3-Clause License. [Link to full license](https://github.com/pytorch/vision/blob/main/LICENSE)
2. mmrotate is licensed under the Apache License 2.0. [Link to full license](https://github.com/open-mmlab/mmrotate/blob/main/LICENSE)

