// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <torch/types.h>

namespace mmrotate {
  at::Tensor nms_rotated_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const double iou_threshold);

  at::Tensor nms_rotated_cuda(
      const at::Tensor& dets,
      const at::Tensor& scores,
      const double iou_threshold, 
      const int multi_label);
  // Interface for Python
  inline at::Tensor nms_rotated(
      const at::Tensor& dets,
      const at::Tensor& scores,
      const double iou_threshold, 
      const int multi_label) {
    TORCH_INTERNAL_ASSERT(dets.device() == scores.device(), "dets and scores must be on same device (GPU | CPU)");
    if (dets.device().is_cuda()) {
      return nms_rotated_cuda(dets.contiguous(), scores.contiguous(), iou_threshold, multi_label);
    }
    return nms_rotated_cpu(dets.contiguous(), scores.contiguous(), iou_threshold);
  }
} // namespace mmrotate