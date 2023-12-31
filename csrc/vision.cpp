// Copyright (c) Facebook, Inc. and its affiliates.

#include <torch/extension.h>
#include "roi_align_rotated/ROIAlignRotated.h"
#include "box_iou_rotated/box_iou_rotated.h"
#include "nms_rotated/nms_rotated.h"

namespace mmrotate {

extern int get_cudart_version();

std::string get_cuda_version() {
  std::ostringstream oss;
  oss << "CUDA ";

  // copied from
  // https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/cuda/detail/CUDAHooks.cpp#L231
  auto printCudaStyleVersion = [&](int v) {
    oss << (v / 1000) << "." << (v / 10 % 100);
    if (v % 10 != 0) {
      oss << "." << (v % 10);
    }
  };
  printCudaStyleVersion(get_cudart_version());
  return oss.str();
}

bool has_cuda() {
  return true;
}

// similar to
// https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/Version.cpp
std::string get_compiler_version() {
  std::ostringstream ss;
#if defined(__GNUC__)
#ifndef __clang__

#if ((__GNUC__ <= 4) && (__GNUC_MINOR__ <= 8))
#error "GCC >= 4.9 is required!"
#endif

  { ss << "GCC " << __GNUC__ << "." << __GNUC_MINOR__; }
#endif
#endif

#if defined(__clang_major__)
  {
    ss << "clang " << __clang_major__ << "." << __clang_minor__ << "."
       << __clang_patchlevel__;
  }
#endif

#if defined(_MSC_VER)
  { ss << "MSVC " << _MSC_FULL_VER; }
#endif
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("nms_rotated", &nms_rotated);
  m.def("box_iou_rotated", &box_iou_rotated);
  m.def("roi_align_rotated_forward", &ROIAlignRotated_forward);
  m.def("roi_align_rotated_backward", &ROIAlignRotated_backward);

  m.def("get_compiler_version", &get_compiler_version, "get_compiler_version");
  m.def("get_cuda_version", &get_cuda_version, "get_cuda_version");
  m.def("has_cuda", &has_cuda, "has_cuda");
}

} // namespace mmrotate
