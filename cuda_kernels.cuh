#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

__device__ float compute_iou(float* boxes1, float* boxes2, int num_boxes);

#endif  // CUDA_KERNELS_CUH
