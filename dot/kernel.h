//
// Created by hzw on 2025/10/25.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_KERNEL_H

#include <float.h>

#include "util/util.h"

// dot
__global__ void dot(unsigned N, float *a, const float *b, float *ret);

__device__ __forceinline__ void shuffle_down_reduce(float &value, const unsigned NUM) {
#pragma unroll
    for (unsigned offset = NUM >> 1; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
}

__device__ __forceinline__ void shuffle_xor_reduce(float &value, const unsigned NUM) {
#pragma unroll
    for (unsigned offset = NUM >> 1; offset > 0; offset >>= 1) {
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
__global__ void dot_warp_shuffle_down(const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tmp[WARP_NUM];
    const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x, strip = gridDim.x * blockDim.x,
            warpIdx = threadIdx.x / warpSize, laneIdx = threadIdx.x - warpIdx * warpSize;
    float value = 0.0f;
    for (unsigned i = idx; i < N; i += strip) {
        value += a[i] * b[i];
    }
    __syncwarp();
    shuffle_down_reduce(value,WARP_SIZE);
    if (laneIdx == 0) {
        tmp[warpIdx] = value;
    }
    __syncthreads();
    if (warpIdx == 0) {
        value = laneIdx < WARP_NUM ? tmp[laneIdx] : 0.0f;
        shuffle_down_reduce(value, WARP_NUM);
        if (laneIdx == 0) {
            atomicAdd(ret, value);
        }
    }
}

template<unsigned WARP_NUM>
__global__ void dot_warp_shuffle_xor_v0(const unsigned N, float *a, float *b, float *ret) {
    __shared__ float tmp[WARP_NUM];
    const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x, strip = gridDim.x * blockDim.x,
            warpIdx = threadIdx.x / warpSize, laneIdx = threadIdx.x - warpIdx * warpSize;
    float value = 0.0f;
    for (unsigned i = idx; i < N; i += strip) {
        value += a[i] * b[i];
    }
    __syncwarp();
    shuffle_xor_reduce(value,WARP_SIZE);
    if (laneIdx == 0) {
        tmp[warpIdx] = value;
    }
    __syncthreads();
    if (warpIdx == 0) {
        value = laneIdx < WARP_NUM ? tmp[laneIdx] : 0.0f;
        shuffle_xor_reduce(value, WARP_NUM);
        if (laneIdx == 0) {
            atomicAdd(ret, value);
        }
    }
}

/**
 *
 *
 * @tparam NUM_WARP block中warp数目
 * @param N 数据维度
 * @param a 输入
 * @param b 输入
 * @param c 输出
 */
template<unsigned NUM_WARP>
__global__ void dot_warp_shuffle_xor_v1(const unsigned N, float *a, float *b, float *c) {
    __shared__ float smem[NUM_WARP];
    const unsigned threadIdxGlobal = blockIdx.x * blockDim.x + threadIdx.x,
            NUM_THREAD = gridDim.x * blockDim.x,
            warpIdx = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    float sum = 0.0f;
    for (unsigned i = threadIdxGlobal; i < N; i += NUM_THREAD) {
        sum += a[i] * b[i];
    }
    __syncwarp();
    shuffle_xor_reduce(sum,WARP_SIZE);
    if (lane == 0) {
        smem[warpIdx] = sum;
    }
    __syncthreads();
    sum = lane < NUM_WARP ? smem[lane] : 0.0f;
    shuffle_xor_reduce(sum, NUM_WARP);
    // sum = __shfl_sync(0xffffffff,sum,0);//broadcast
    if (threadIdx.x == 0) {
        atomicAdd(c, sum);
    }
}
#endif // CUDA_TUTORIAL_EXAMPLE_KERNEL_H
