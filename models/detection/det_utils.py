# Adapted from torchvision/models/detection/_utils.py 
import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, FrozenBatchNorm2d, generalized_box_iou_loss

from .box_coders import BoxCoder, HBoxCoder, OBoxCoder