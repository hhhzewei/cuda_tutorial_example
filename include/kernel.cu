//
// Created by hzw on 2025/10/25.
//
// Kernel function to add the elements of two arrays
#include <cstdio>

#define FLOAT4(x) (*((float4*)(&x)))
#define CEIL(a,b) ((a+b-1)/b)

__global__ void add(unsigned n, float *x, float *y, float *ret) {
    unsigned int strip = gridDim.x * blockDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += strip)
        ret[i] = x[i] + y[i];
}

__global__ void add_float4(unsigned N, float *a, float *b, float *ret) {
    for (unsigned int i = (threadIdx.x + blockDim.x * blockIdx.x) * 4; i < N; i += blockDim.x * gridDim.x * 4) {
        float4 tmp_a = FLOAT4(a[i]), tmp_b = FLOAT4(b[i]), tmp_ret;
        tmp_ret.x = tmp_a.x + tmp_b.x;
        tmp_ret.y = tmp_a.y + tmp_b.y;
        tmp_ret.z = tmp_a.z + tmp_b.z;
        tmp_ret.w = tmp_a.w + tmp_b.w;
        FLOAT4(ret[i]) = tmp_ret;
    }
}

__global__ void dot(unsigned N, float *a, float *b, float *ret) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int strip = gridDim.x * blockDim.x;
    a[idx] = a[idx] * b[idx];
    for (unsigned int i = idx + strip; i < N; i += strip) {
        a[idx] += a[i] * b[i];
    }
    __syncthreads();
    unsigned int range = blockDim.x >> 1;
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

__global__ void dot_shared_external(unsigned N, float *a, float *b, float *ret) {
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


__global__ void transpose_naive(unsigned m, unsigned N, float *input, float *output) {
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < N && y < N) {
        output[x * m + y] = input[y * N + x];
    }
}

// sgemm
__global__ void sgemm_naive(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M) {
        float value = 0.0f;
        for (unsigned i = 0; i < K; ++i) {
            value += a[y * K + i] * b[i * N + x];
        }
        ret[y * N + x] = value;
    }
}
