//
// Created by hzw on 2025/10/25.
//
// Kernel function to add the elements of two arrays
#include "kernel.h"

__global__ void dot(const unsigned N, float *a, const float *b, float *ret) {
    const unsigned idx = threadIdx.x + blockDim.x * blockIdx.x,
            tid = threadIdx.x,
            strip = gridDim.x * blockDim.x;
    a[idx] = a[idx] * b[idx];
    for (unsigned i = idx + strip; i < N; i += strip) {
        a[idx] += a[i] * b[i];
    }
    __syncthreads();
    unsigned range = blockDim.x >> 1;
    while (range > 0) {
        if (tid < range) {
            a[idx] += a[idx + range];
        }
        range >>= 1;
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(ret, a[idx]);
    }
}

__global__ void dot_shared_external(const unsigned N, const float *a, const float *b, float *ret) {
    extern __shared__ float tmp[];
    const unsigned tid = threadIdx.x, idx = tid + blockIdx.x * blockDim.x, strip = gridDim.x * blockDim.x;
    tmp[tid] = 0;
    for (unsigned i = idx; i < N; i += strip) {
        tmp[tid] += a[i] * b[i];
    }
    __syncthreads();
    unsigned int range = blockDim.x >> 1;
    while (range > 0) {
        if (tid < range) {
            tmp[tid] += tmp[tid + range];
        }
        range >>= 1;
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(ret, tmp[0]);
    }
}
