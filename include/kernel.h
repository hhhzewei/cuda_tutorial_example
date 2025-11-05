//
// Created by hzw on 2025/10/25.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_KERNEL_H

#define CEIL(a,b) (((a)+(b)-1)/(b))


// add
__global__ void add(unsigned n, float *a, float *b, float *ret);

/**
 *
 * @param n % 4 == 0
 * @param a
 * @param b
 * @param ret
 */
__global__ void add_float4(unsigned n, float *a, float *b, float *ret);

// dot
__global__ void dot(unsigned n, float *a, float *b, float *ret);

template<const int BLOCK_DIM>
__global__ void dot_share(unsigned n, float *a, float *b, float *ret) {
    __shared__ float tmp[BLOCK_DIM];
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    tmp[tid] = 0;
    unsigned int strip = gridDim.x * blockDim.x;
    for (unsigned int i = idx; i < n; i += strip) {
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
        atomicAdd(ret, tmp[tid]);
    }
}

__global__ void dot_shared_external(unsigned n, float *a, float *b, float *ret);

/**
 *
 * blockDim <= warpSize * warpSize
 */
__global__ void dot_warp_shuffle(unsigned n, float *a, float *b, float *ret);

// transport
__global__ void transpose_naive(unsigned m, unsigned n, float *input, float *output);

template<unsigned WARP_SIZE>
__global__ void transpose_padding(unsigned m, unsigned n, float *input, float *output) {
    // padding
    __shared__ float tile[WARP_SIZE][WARP_SIZE + 1];
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
            y = blockIdx.y * blockDim.y + threadIdx.y,
            tx = threadIdx.x,
            ty = threadIdx.y;
    if (x < n && y < m) {
        tile[ty][tx] = input[y * n + x];
    }
    __syncthreads();
    unsigned x1 = blockDim.x * blockIdx.x + ty, y1 = blockDim.y * blockIdx.y + tx;
    if (x1 < n && y1 < m) {
        output[x1 * m + y1] = tile[tx][ty];
    }
}

template<unsigned WARP_SIZE>
__global__ void transpose_swizzle(unsigned m, unsigned n, float *input, float *output) {
    // padding
    __shared__ float tile[WARP_SIZE][WARP_SIZE];
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
            y = blockIdx.y * blockDim.y + threadIdx.y,
            tx = threadIdx.x,
            ty = threadIdx.y;
    if (x < n && y < m) {
        tile[ty][tx ^ ty] = input[y * n + x];
    }
    __syncthreads();
    unsigned x1 = blockDim.x * blockIdx.x + ty, y1 = blockDim.y * blockIdx.y + tx;
    if (x1 < n && y1 < m) {
        output[x1 * m + y1] = tile[tx][ty ^ tx];
    }
}

// sgemm
__global__ void sgemm_naive(unsigned m, unsigned k, unsigned n, float *a, float *b, float *ret);

template<unsigned TILE_M, unsigned TILE_N, unsigned TILE_K>
__global__ void sgemm_block_tile(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    __shared__ unsigned tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    unsigned tx = threadIdx.x, ty = threadIdx.y,
            x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned value = 0.0f;
    for (unsigned k = 0; k < K; k += TILE_K) {
        tile_a[ty][tx] = a[y * K + k + tx];
        tile_b[ty][tx] = b[(k + ty) * N + x];
        __syncthreads();
        for (unsigned tk = 0; tk < TILE_K; ++tk) {
            value += tile_a[ty][tk] * tile_b[tk][tx];
        }
        __syncthreads();
    }
    ret[y * N + x] = value;
}

template<unsigned TILE_M, unsigned TILE_N, unsigned TILE_K,  unsigned THREAD_M, unsigned THREAD_N>
__global__ void sgemm_thread_tile(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    __shared__ unsigned tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    unsigned tx = threadIdx.x * THREAD_N, ty = threadIdx.y * THREAD_M,
            x = (blockDim.x * blockIdx.x + threadIdx.x) * THREAD_N, y =
                    (blockDim.y * blockIdx.y + threadIdx.y) * THREAD_M;
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    for (unsigned k = 0; k < K; k += TILE_K) {
        // 填充共享内存
        for (unsigned i = 0; i < THREAD_M; ++i) {
            for (unsigned j = 0; j < THREAD_N; ++j) {
                tile_a[ty + i][tx + j] = a[(y + i) * K + (k + tx + j)];
                tile_b[ty + i][tx + j] = b[(k + ty + i) * N + (x + j)];
            }
        }
        __syncthreads();
        // 填充寄存器
        float thread_tile_a[THREAD_M][TILE_K] = {0.0f}, thread_tile_b[TILE_K][THREAD_N] = {0.0f};
        for (unsigned i = 0; i < THREAD_M; ++i) {
            for (unsigned j = 0; j < TILE_K; ++j) {
                thread_tile_a[i][j] = tile_a[ty + i][j];
            }
        }
        for (unsigned i = 0; i < TILE_K; ++i) {
            for (unsigned j = 0; j < THREAD_N; ++j) {
                thread_tile_b[i][j] = tile_b[i][tx + j];
            }
        }
        // 计算结果
        for (unsigned i = 0; i < THREAD_M; ++i) {
            for (unsigned j = 0; j < THREAD_N; j++) {
                for (unsigned tk = 0; tk < TILE_K; ++tk) {
                    ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
                }
            }
        }
        __syncthreads();
    }
    for (unsigned i = 0; i < THREAD_M; ++i) {
        for (unsigned j = 0; j < THREAD_N; j++) {
            ret[(y + i) * N + x + j] = ret_tile[i][j];
        }
    }
}


#endif // CUDA_TUTORIAL_EXAMPLE_KERNEL_H
