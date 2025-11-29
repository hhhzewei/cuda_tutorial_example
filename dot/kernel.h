//
// Created by hzw on 2025/10/25.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_KERNEL_H

#include "util/util.h"

// dot
__global__ void dot(unsigned N, float *a, const float *b, float *ret);

__device__ __forceinline__ void reduce_add_v0(float &value) {
#pragma unroll
    for (unsigned offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
}

__device__ __forceinline__ void reduce_add_v1(float &value) {
#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
}

template<const int BLOCK_DIM>
__global__ void dot_share(const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tmp[BLOCK_DIM];
    const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x,
            tid = threadIdx.x;
    tmp[tid] = 0;
    const unsigned strip = gridDim.x * blockDim.x;
    for (unsigned i = idx; i < N; i += strip) {
        tmp[tid] += a[i] * b[i];
    }
    __syncthreads();
    unsigned range = blockDim.x >> 1;
    while (range > 0) {
        if (tid < range) {
            tmp[tid] += tmp[tid + range];
        }
        range >>= 1;
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(ret, tmp[tid]);
    }
}

__global__ void dot_shared_external(unsigned N, const float *a, const float *b, float *ret);

/**
 *
 * warp_num = blockDim / warpSize <=  * warpSize
 */
template<unsigned WARP_NUM>
__global__ void dot_warp_shuffle_v0(const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tmp[WARP_NUM];
    const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x, strip = gridDim.x * blockDim.x,
            warpNum = CEIL(blockDim.x, warpSize), warpIdx = threadIdx.x / warpSize, laneIdx =
                    threadIdx.x - warpIdx * warpSize;
    float value = 0.0f;
    for (unsigned i = idx; i < N; i += strip) {
        value += a[i] * b[i];
    }
    __syncwarp();
    reduce_add_v0(value);
    if (laneIdx == 0) {
        tmp[warpIdx] = value;
    }
    __syncthreads();
    if (warpIdx == 0) {
        value = laneIdx < warpNum ? tmp[laneIdx] : 0.0f;
        reduce_add_v0(value);
        if (laneIdx == 0) {
            atomicAdd(ret, value);
        }
    }
}

template<unsigned WARP_NUM>
__global__ void dot_warp_shuffle_v1(const unsigned N, float *a, float *b, float *ret) {
    __shared__ float tmp[WARP_NUM];
    const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x, strip = gridDim.x * blockDim.x,
            warpNum = CEIL(blockDim.x, warpSize), warpIdx = threadIdx.x / warpSize, laneIdx =
                    threadIdx.x - warpIdx * warpSize;
    float value = 0.0f;
    for (unsigned i = idx; i < N; i += strip) {
        value += a[i] * b[i];
    }
    __syncwarp();
    reduce_add_v1(value);
    if (laneIdx == 0) {
        tmp[warpIdx] = value;
    }
    __syncthreads();
    if (warpIdx == 0) {
        value = laneIdx < warpNum ? tmp[laneIdx] : 0.0f;
        reduce_add_v1(value);
        if (laneIdx == 0) {
            atomicAdd(ret, value);
        }
    }
}
#endif // CUDA_TUTORIAL_EXAMPLE_KERNEL_H
