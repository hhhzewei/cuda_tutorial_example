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

__global__ void add_float4(unsigned n, float *a, float *b, float *ret) {
    for (unsigned int i = (threadIdx.x + blockDim.x * blockIdx.x) * 4; i < n; i += blockDim.x * gridDim.x * 4) {
        float4 tmp_a = FLOAT4(a[i]), tmp_b = FLOAT4(b[i]), tmp_ret;
        tmp_ret.x = tmp_a.x + tmp_b.x;
        tmp_ret.y = tmp_a.y + tmp_b.y;
        tmp_ret.z = tmp_a.z + tmp_b.z;
        tmp_ret.w = tmp_a.w + tmp_b.w;
        FLOAT4(ret[i]) = tmp_ret;
    }
}

__global__ void dot(unsigned n, float *a, float *b, float *ret) {
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int strip = gridDim.x * blockDim.x;
    a[idx] = a[idx] * b[idx];
    for (unsigned int i = idx + strip; i < n; i += strip) {
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

__global__ void dot_shared_external(unsigned n, float *a, float *b, float *ret) {
    extern __shared__ float tmp[];
    const unsigned tid = threadIdx.x, idx = tid + blockIdx.x * blockDim.x, strip = gridDim.x * blockDim.x;
    tmp[tid] = 0;
    for (unsigned i = idx; i < n; i += strip) {
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

__global__ void dot_warp_shuffle(unsigned n, float *a, float *b, float *ret) {
    extern __shared__ float tmp[];
    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x, strip = gridDim.x * blockDim.x;
    unsigned warpNum = CEIL(blockDim.x, warpSize), warpIdx = threadIdx.x / warpSize, laneIdx =
            threadIdx.x - warpIdx * warpSize;
    if (laneIdx == 0) {
        tmp[warpIdx] = 0.0f;
    }
    float value = 0.0f;
    for (unsigned i = idx; i < n; i += strip) {
        value += a[i] * b[i];
    }
    __syncwarp();
    unsigned range = warpSize >> 1;
    while (range) {
        value += __shfl_down_sync(0xffffffff, value, range);
        range >>= 1;
    }
    if (laneIdx == 0) {
        tmp[warpIdx] = value;
    }
    __syncthreads();
    if (warpIdx == 0) {
        range = warpNum >> 1;
        value = laneIdx < warpNum ? tmp[laneIdx] : 0.0f;
        while (range) {
            value += __shfl_down_sync(0xffffffff, value, range);
            range >>= 1;
        }
        if (laneIdx == 0) {
            atomicAdd(ret, value);
        }
    }
}

__global__ void transpose_naive(unsigned m, unsigned n, float *input, float *output) {
    unsigned x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < n && y < n) {
        output[x * m + y] = input[y * n + x];
    }
}

// sgemm
__global__ void sgemm_naive(unsigned m, unsigned k, unsigned n, float *a, float *b, float *ret) {
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < m) {
        float value = 0.0f;
        for (unsigned i = 0; i < k; ++i) {
            value += a[y * k + i] * b[i * n + x];
        }
        ret[y * n + x] = value;
    }
}
