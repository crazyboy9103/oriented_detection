#include "cuda_kernels.cuh"

__global__ void compute_iou_kernel(float* boxes1, float* boxes2, float* ious, int num_boxes) {
    // CUDA kernel implementation for IOU computation
}

extern "C" float compute_iou(float* boxes1, float* boxes2, int num_boxes) {
    // Allocate memory on GPU
    float* d_boxes1;
    float* d_boxes2;
    float* d_ious;
    cudaMalloc((void**)&d_boxes1, num_boxes * sizeof(float));
    cudaMalloc((void**)&d_boxes2, num_boxes * sizeof(float));
    cudaMalloc((void**)&d_ious, num_boxes * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_boxes1, boxes1, num_boxes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_boxes2, boxes2, num_boxes * sizeof(float), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    compute_iou_kernel<<<1, 1>>>(d_boxes1, d_boxes2, d_ious, num_boxes);

    // Copy result from GPU to CPU
    float iou;
    cudaMemcpy(&iou, d_ious, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_boxes1);
    cudaFree(d_boxes2);
    cudaFree(d_ious);

    return iou;
}
