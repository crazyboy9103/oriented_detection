# Optional 
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
# Compile csrc
pip install -e .

```python
import torch # torch must be imported before detectron2
from detectron2._C import (
    get_compiler_version,
    get_cuda_version,
    has_cuda,
    # deform_conv_forward,
    # deform_conv_backward_input,
    # deform_conv_backward_filter,
    # modulated_deform_conv_forward,
    # modulated_deform_conv_backward,
    COCOevalAccumulate,
    COCOevalEvaluateImages,
    InstanceAnnotation,
    ImageEvaluation,

    # functions defined in torch.ops.detectron2 in original detectron2
    nms_rotated,
    box_iou_rotated,
    roi_align_rotated_forward,
    roi_align_rotated_backward
)

```