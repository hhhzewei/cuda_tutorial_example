//
// Created by hzw on 2025/10/25.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_KERNEL_H

// add
__global__ void add(unsigned N, const float *a, const float *b, float *ret);

/**
 *
 * @param N % 4 == 0
 * @param a
 * @param b
 * @param ret
 */
__global__ void add_float4(unsigned N, float *a, float *b, float *ret);

#endif // CUDA_TUTORIAL_EXAMPLE_KERNEL_H
