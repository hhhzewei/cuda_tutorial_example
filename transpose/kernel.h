//
// Created by hzw on 2025/10/25.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_KERNEL_H

__global__ void transpose_naive(unsigned M, unsigned N, const float *input, float *output);

__global__ void transpose_shared(unsigned M, unsigned N, float *input, float *output);

__global__ void transpose_padding(unsigned M, unsigned N, float *input, float *output);

__global__ void transpose_swizzle(unsigned M, unsigned N, float *input, float *output);

#endif // CUDA_TUTORIAL_EXAMPLE_KERNEL_H
