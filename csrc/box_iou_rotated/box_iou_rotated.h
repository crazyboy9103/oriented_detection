// Copyright (c) Facebook, Inc. and its affiliates.
#pragma once
#include <torch/types.h>

namespace detectron2 {

at::Tensor box_iou_rotated_cpu(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    bool angle_aware
);

#if defined(WITH_CUDA) || defined(WITH_HIP)
at::Tensor box_iou_rotated_cuda(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    bool angle_aware   
);
#endif

// Interface for Python
// inline is needed to prevent multiple function definitions when this header is
// included by different cpps
// angle_aware parameter modulates ious following this paper: "Learning a Rotation Invariant Detector with Rotatable Bounding Box" Lei Liu et al.
inline at::Tensor box_iou_rotated(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    bool angle_aware) {
  assert(boxes1.device().is_cuda() == boxes2.device().is_cuda());
  if (boxes1.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return box_iou_rotated_cuda(boxes1.contiguous(), boxes2.contiguous(), angle_aware);
#else
    AT_ERROR("Detectron2 is not compiled with GPU support!");
#endif
  }

  return box_iou_rotated_cpu(boxes1.contiguous(), boxes2.contiguous(), angle_aware);
}

} // namespace detectron2