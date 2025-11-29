//
// Created by hzw on 2025/10/25.
//
// Kernel function to add the elements of two arrays
#include "kernel.h"
#include "util/util.h"

__global__ void transpose_naive(const unsigned M, const unsigned N, const float *input, float *output) {
    const unsigned x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < N && y < N) {
        output[x * M + y] = input[y * N + x];
    }
}

__global__ void transpose_shared(const unsigned M, const unsigned N, const float *input, float *output) {
    __shared__ float tile[WARP_SIZE][WARP_SIZE];
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
            y = blockIdx.y * blockDim.y + threadIdx.y,
            tx = threadIdx.x,
            ty = threadIdx.y;
    if (x < N && y < M) {
        tile[ty][tx] = input[y * N + x];
    }
    __syncthreads();
    const unsigned x1 = blockDim.x * blockIdx.x + ty, y1 = blockDim.y * blockIdx.y + tx;
    if (x1 < N && y1 < M) {
        output[x1 * M + y1] = tile[tx][ty];
    }
}

__global__ void transpose_padding(const unsigned M, const unsigned N, const float *input, float *output) {
    // padding
    __shared__ float tile[WARP_SIZE][WARP_SIZE + 1];
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
            y = blockIdx.y * blockDim.y + threadIdx.y,
            tx = threadIdx.x,
            ty = threadIdx.y;
    if (x < N && y < M) {
        tile[ty][tx] = input[y * N + x];
    }
    __syncthreads();
    const unsigned x1 = blockDim.x * blockIdx.x + ty, y1 = blockDim.y * blockIdx.y + tx;
    if (x1 < N && y1 < M) {
        output[x1 * M + y1] = tile[tx][ty];
    }
}

__global__ void transpose_swizzle(const unsigned M, const unsigned N, const float *input, float *output) {
    // padding
    __shared__ float tile[WARP_SIZE][WARP_SIZE];
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
            y = blockIdx.y * blockDim.y + threadIdx.y,
            tx = threadIdx.x,
            ty = threadIdx.y;
    if (x < N && y < M) {
        tile[ty][tx ^ ty] = input[y * N + x];
    }
    __syncthreads();
    const unsigned x1 = blockDim.x * blockIdx.x + ty, y1 = blockDim.y * blockIdx.y + tx;
    if (x1 < N && y1 < M) {
        output[x1 * M + y1] = tile[tx][ty ^ tx];
    }
}
