# Optional 
export PATH=/usr/local/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
# Compile csrc
python setup.py install 
# or 
pip install -e .
```python
import torch # torch must be imported before odtk
from odtk._C import decode, iou, nms 
```