# Recommended: Dockerfile
```shell
docker build -t ${DOCKER IMAGE TAG}.
docker run -it -e WANDB_API_KEY="${WANDB API KEY}" --gpus all -v ${HOST PATH TO DATASET FOLDER}:/datasets --shm-size=40G --name ${CONTAINER NAME} rot-det 
```
# Compile csrc
```shell
pip install -e .
```

```python
import torch # torch must be imported before detectron2
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