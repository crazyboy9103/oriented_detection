// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
// modified from
// https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/csrc/box_iou_rotated/box_iou_rotated_cuda.cu
#include "box_iou_rotated_cuda.cuh"
#include "../pytorch_cuda_helper.hpp"

namespace mmrotate {
    at::Tensor box_iou_rotated_cuda(
        const Tensor& boxes1, 
        const Tensor& boxes2, 
        const int mode_flag, 
        const bool aligned
    ) {
    using scalar_t = float;
    TORCH_INTERNAL_ASSERT(boxes1.is_cuda(), "boxes1 must be a CUDA tensor");
    TORCH_INTERNAL_ASSERT(boxes2.is_cuda(), "boxes2 must be a CUDA tensor");

    int num_boxes1 = boxes1.size(0);
    int num_boxes2 = boxes2.size(0);
    int output_size = num_boxes1 * num_boxes2;

    at::Tensor ious = at::empty({num_boxes1 * num_boxes2}, boxes1.options().dtype(at::kFloat));

    at::cuda::CUDAGuard device_guard(boxes1.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    box_iou_rotated_cuda_kernel<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, stream>>>(
            num_boxes1, 
            num_boxes2, 
            boxes1.data_ptr<scalar_t>(),
            boxes2.data_ptr<scalar_t>(), 
            (scalar_t*)ious.data_ptr<scalar_t>(),
            mode_flag, 
            aligned
        );
    AT_CUDA_CHECK(cudaGetLastError());

    // reshape from 1d array to 2d array
    auto shape = std::vector<int64_t>{num_boxes1, num_boxes2};
    return ious.view(shape);
    }
} // namespace mmrotate