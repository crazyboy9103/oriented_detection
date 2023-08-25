// Copyright (c) Facebook, Inc. and its affiliates.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include "box_iou_rotated_utils.h"


namespace detectron2 {

// 2D block with 32 * 16 = 512 threads per block
const int BLOCK_DIM_X = 32;
const int BLOCK_DIM_Y = 16;

template <typename T>
__global__ void box_iou_rotated_cuda_kernel(
    const int n_boxes1,
    const int n_boxes2,
    const T* dev_boxes1,
    const T* dev_boxes2,
    T* dev_ious, 
    const bool aligned) {
  if (aligned) {
    CUDA_1D_KERNEL_LOOP(index, n_boxes1) {
      int b1 = index;
      int b2 = index;

      int base1 = b1 * 5;

      float block_boxes1[5];
      float block_boxes2[5];

      block_boxes1[0] = dev_boxes1[base1 + 0];
      block_boxes1[1] = dev_boxes1[base1 + 1];
      block_boxes1[2] = dev_boxes1[base1 + 2];
      block_boxes1[3] = dev_boxes1[base1 + 3];
      block_boxes1[4] = dev_boxes1[base1 + 4];

      int base2 = b2 * 5;

      block_boxes2[0] = dev_boxes2[base2 + 0];
      block_boxes2[1] = dev_boxes2[base2 + 1];
      block_boxes2[2] = dev_boxes2[base2 + 2];
      block_boxes2[3] = dev_boxes2[base2 + 3];
      block_boxes2[4] = dev_boxes2[base2 + 4];

      dev_ious[index] =
          single_box_iou_rotated<T>(block_boxes1, block_boxes2);
    }
  } else {
    CUDA_1D_KERNEL_LOOP(index, n_boxes1 * n_boxes2) {
      int b1 = index / n_boxes2;
      int b2 = index % n_boxes2;

      int base1 = b1 * 5;

      float block_boxes1[5];
      float block_boxes2[5];

      block_boxes1[0] = dev_boxes1[base1 + 0];
      block_boxes1[1] = dev_boxes1[base1 + 1];
      block_boxes1[2] = dev_boxes1[base1 + 2];
      block_boxes1[3] = dev_boxes1[base1 + 3];
      block_boxes1[4] = dev_boxes1[base1 + 4];

      int base2 = b2 * 5;

      block_boxes2[0] = dev_boxes2[base2 + 0];
      block_boxes2[1] = dev_boxes2[base2 + 1];
      block_boxes2[2] = dev_boxes2[base2 + 2];
      block_boxes2[3] = dev_boxes2[base2 + 3];
      block_boxes2[4] = dev_boxes2[base2 + 4];

      dev_ious[index] =
          single_box_iou_rotated<T>(block_boxes1, block_boxes2);
      }
  }
}

at::Tensor box_iou_rotated_cuda(
    // input must be contiguous
    const at::Tensor& boxes1,
    const at::Tensor& boxes2) {
  using scalar_t = float;
  AT_ASSERTM(boxes1.scalar_type() == at::kFloat, "boxes1 must be a float tensor");
  AT_ASSERTM(boxes2.scalar_type() == at::kFloat, "boxes2 must be a float tensor");
  AT_ASSERTM(boxes1.is_cuda(), "boxes1 must be a CUDA tensor");
  AT_ASSERTM(boxes2.is_cuda(), "boxes2 must be a CUDA tensor");
  at::cuda::CUDAGuard device_guard(boxes1.device());

  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  at::Tensor ious =
      at::empty({num_boxes1 * num_boxes2}, boxes1.options().dtype(at::kFloat));
  auto output_size = ious.numel();

  bool transpose = false;
  if (num_boxes1 > 0 && num_boxes2 > 0) {
    scalar_t *data1 = boxes1.data_ptr<scalar_t>(),
             *data2 = boxes2.data_ptr<scalar_t>();

    if (num_boxes2 > 65535 * BLOCK_DIM_Y) {
      AT_ASSERTM(
          num_boxes1 <= 65535 * BLOCK_DIM_Y,
          "Too many boxes for box_iou_rotated_cuda!");
      // x dim is allowed to be large, but y dim cannot,
      // so we transpose the two to avoid "invalid configuration argument"
      // error. We assume one of them is small. Otherwise the result is hard to
      // fit in memory anyway.
      std::swap(num_boxes1, num_boxes2);
      std::swap(data1, data2);
      transpose = true;
    }

    // const int blocks_x =
    //     at::cuda::ATenCeilDiv(static_cast<int>(num_boxes1), BLOCK_DIM_X);
    // const int blocks_y =
    //     at::cuda::ATenCeilDiv(static_cast<int>(num_boxes2), BLOCK_DIM_Y);

    // dim3 blocks(blocks_x, blocks_y);
    // dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);
    at::cuda::CUDAGuard device_guard(boxes1.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // box_iou_rotated_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
    //     num_boxes1,
    //     num_boxes2,
    //     data1,
    //     data2,
    //     (scalar_t*)ious.data_ptr<scalar_t>());
    bool aligned = true;
    box_iou_rotated_cuda_kernel<scalar_t>
      <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
          num_boxes1, num_boxes2, boxes1.data_ptr<scalar_t>(),
          boxes2.data_ptr<scalar_t>(), (scalar_t*)ious.data_ptr<scalar_t>(),
          aligned);
    AT_CUDA_CHECK(cudaGetLastError());
  }

  // reshape from 1d array to 2d array
  auto shape = std::vector<int64_t>{num_boxes1, num_boxes2};
  if (transpose) {
    return ious.view(shape).t();
  } else {
    return ious.view(shape);
  }
}

} // namespace detectron2
