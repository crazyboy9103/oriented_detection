// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <torch/types.h>

namespace mmrotate {
  at::Tensor box_iou_rotated_cpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    const int mode_flag,
    const bool aligned
    );

  at::Tensor box_iou_rotated_cuda(
      const at::Tensor& boxes1,
      const at::Tensor& boxes2,
      const int mode_flag, 
      const bool aligned
    );

  // Interface for Python
  inline at::Tensor box_iou_rotated(
      const at::Tensor& boxes1,
      const at::Tensor& boxes2,
      const int mode_flag, 
      const bool aligned
    ) {
    TORCH_INTERNAL_ASSERT(boxes1.device() == boxes2.device(), "boxes1 and boxes2 must be on same device (GPU | CPU)");
    if (boxes1.device().is_cuda()) {
      return box_iou_rotated_cuda(boxes1.contiguous(), boxes2.contiguous(), mode_flag, aligned);
    }
    return box_iou_rotated_cpu(boxes1.contiguous(), boxes2.contiguous(), mode_flag, aligned);
  }
} // namespace mmrotate