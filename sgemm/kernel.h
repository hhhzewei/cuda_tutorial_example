//
// Created by hzw on 2025/10/25.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_KERNEL_H

#include <cuda_pipeline.h>
#include <mma.h>

#include "util/util.h"

// sgemm
__global__ void sgemm_naive(unsigned M, unsigned K, unsigned N, const float *a, const float *b, float *ret);

template<unsigned TILE_M, unsigned TILE_N, unsigned TILE_K>
__global__ void sgemm_block_tile(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    const unsigned tx = threadIdx.x, ty = threadIdx.y,
            x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    float value = 0.0f;
    for (unsigned k = 0; k < K; k += TILE_K) {
        tile_a[ty][tx] = a[y * K + k + tx];
        tile_b[ty][tx] = b[(k + ty) * N + x];
        __syncthreads();
#pragma unroll
        for (unsigned tk = 0; tk < TILE_K; ++tk) {
            value += tile_a[ty][tk] * tile_b[tk][tx];
        }
        __syncthreads();
    }
    ret[y * N + x] = value;
}

template<unsigned TILE_M, unsigned TILE_N, unsigned TILE_K, unsigned THREAD_M, unsigned THREAD_N>
__global__ void sgemm_thread_tile_v0(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    const unsigned tx = threadIdx.x * THREAD_N, ty = threadIdx.y * THREAD_M,
            x = (blockDim.x * blockIdx.x + threadIdx.x) * THREAD_N, y =
                    (blockDim.y * blockIdx.y + threadIdx.y) * THREAD_M;
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    for (unsigned k = 0; k < K; k += TILE_K) {
        // 填充共享内存
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
                tile_a[ty + i][tx + j] = a[(y + i) * K + (k + tx + j)];
                tile_b[ty + i][tx + j] = b[(k + ty + i) * N + (x + j)];
            }
        }
        __syncthreads();
        // 填充寄存器
        float thread_tile_a[THREAD_M][TILE_K] = {0.0f}, thread_tile_b[TILE_K][THREAD_N] = {0.0f};
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < TILE_K; ++j) {
                thread_tile_a[i][j] = tile_a[ty + i][j];
            }
        }
#pragma unroll
        for (unsigned i = 0; i < TILE_K; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
                thread_tile_b[i][j] = tile_b[i][tx + j];
            }
        }
        // 计算结果
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; j++) {
#pragma unroll
                for (unsigned tk = 0; tk < TILE_K; ++tk) {
                    ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; j++) {
            ret[(y + i) * N + x + j] = ret_tile[i][j];
        }
    }
}

/**
 * 2D-strip-loop读全局内存
 *
 * 同warp线程访问b矩阵同一行会bank conflict，如THREAD_N=2时，访问bank序列为0,2,...,30,0,2,...,30
 *
 * 添加thread tile内x方向的offset=tx/warp_circle，swizzle错开bank
 *
 * 惊天负优化
 *
 * @tparam BLOCK_M blockDim.y
 * @tparam BLOCK_N blockDim.x
 * @tparam TILE_K K方向取共享内存TILE尺寸
 * @tparam THREAD_M 寄存器TILE
 * @tparam THREAD_N 寄存器TILE
 * @tparam WARP_CIRCLE_LOG $log_2(warp_size/THREAD_N)$，
 *
 */
template<unsigned BLOCK_M, unsigned BLOCK_N,
    unsigned TILE_K,
    unsigned THREAD_M, unsigned THREAD_N, unsigned
    WARP_CIRCLE_LOG>
__global__ void sgemm_thread_tile_v1(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    constexpr unsigned TILE_M = BLOCK_M * THREAD_M, TILE_N = BLOCK_N * THREAD_N;
    __shared__ float tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    const unsigned tx = threadIdx.x, ty = threadIdx.y,
            x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    // unsigned warp_circle = CEIL(warpSize, THREAD_N);
    // offset=tx/warp_circle
    const unsigned offset = tx >> WARP_CIRCLE_LOG;
    constexpr unsigned mask = THREAD_N - 1;
    for (unsigned k = 0; k < K; k += TILE_K) {
        // 填充共享内存，strip loop
        for (unsigned i = ty; i < TILE_M; i += blockDim.y) {
            for (unsigned j = tx; j < TILE_K; j += blockDim.x) {
                tile_a[i][j] = a[(i + blockIdx.y * TILE_M) * K + (j + k)];
            }
        }
        for (unsigned i = ty; i < TILE_K; i += blockDim.y) {
            for (unsigned j = tx; j < TILE_N; j += blockDim.x) {
                tile_b[i][j] = b[(i + k) * N + (j + blockIdx.x * TILE_N)];
            }
        }
        __syncthreads();
        // 填充寄存器
        float thread_tile_a[THREAD_M][TILE_K] = {0.0f}, thread_tile_b[TILE_K][THREAD_N] = {0.0f};
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < TILE_K; ++j) {
                thread_tile_a[i][j] = tile_a[ty * THREAD_M + i][j];
            }
        }
        // 一个线程x方向负责THREAD_N个元素，读共享内存会bank_conflict
        // 线程[0~warp_circle)为一组不重复bank，然后下一组重复bank
#pragma unroll
        for (unsigned i = 0; i < TILE_K; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
                // warp错开
                // j_offset=(j_offset)%thread_n
                thread_tile_b[i][j + offset & mask] = tile_b[i][tx * THREAD_N + (j + offset & mask)];
                // if (blockIdx.x == 0 && blockIdx.y == 0 && ty == 0 && k == 0 && i < 4) {
                //     printf("tx:%u, ty:%u,i:%u,j:%u,bank:%u,offset bank:%u,warpSize:%u\n",
                //            tx, ty,
                //            i, j,
                //            (tx * THREAD_N + j) % warpSize, (tx * THREAD_N + j_offset) % warpSize,
                //            warpSize);
                // }
                // thread_tile_b[i][j] = tile_b[i][tx * THREAD_N + j];
            }
        }
        // 计算结果
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; j++) {
#pragma unroll
                for (unsigned tk = 0; tk < TILE_K; ++tk) {
                    ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; j++) {
            ret[(y * THREAD_M + i) * N + x * THREAD_N + j] = ret_tile[i][j];
        }
    }
}


/**
 * 2D-strip-loop读全局内存
 *
 * padding 解决bank conflict
 *
 * @tparam TILE_M TILE尺寸，TILE_M==BLOCK_M*THREAD_M
 * @tparam TILE_K TILE尺寸，TILE_K==WARP_SIZE整数倍
 * @tparam TILE_N TILE尺寸，TILE_N==BLOCK_N*THREAD_N==WARP_SIZE整数倍
 * @tparam THREAD_M
 * @tparam THREAD_N
 */
template<unsigned TILE_M = 32, unsigned TILE_K = 32, unsigned TILE_N = 32,
    unsigned THREAD_M = 2, unsigned THREAD_N = 2>
__global__ void sgemm_thread_tile_v2(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    // padding
    // 一个线程x方向负责THREAD_N个元素，读共享内存会bank_conflict
    __shared__ float tile_a[TILE_M][TILE_K + 1], tile_b[TILE_K][TILE_N + 1];
    const unsigned tx = threadIdx.x, ty = threadIdx.y, tIdx = ty * blockDim.x + tx,
            x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y,
            warpNum = blockDim.x * blockDim.y / warpSize
            , warpIdx = tIdx / warpSize, laneIdx = tIdx - warpIdx * warpSize;
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    for (unsigned k = 0; k < K; k += TILE_K) {
        // 填充共享内存，strip loop
        for (unsigned i = warpIdx; i < TILE_M; i += warpNum) {
            for (unsigned j = laneIdx; j < TILE_K; j += warpSize) {
                // a[i + blockIdx.y * TILE_M][j+k]
                tile_a[i][j] = a[(i + blockIdx.y * TILE_M) * K + (j + k)];
            }
        }
        for (unsigned i = warpIdx; i < TILE_K; i += warpNum) {
            for (unsigned j = laneIdx; j < TILE_N; j += warpSize) {
                // b[i+k][j + blockIdx.x * TILE_N]
                tile_b[i][j] = b[(i + k) * N + (j + blockIdx.x * TILE_N)];
            }
        }
        __syncthreads();
        // 填充寄存器
        float thread_tile_a[THREAD_M][TILE_K] = {0.0f}, thread_tile_b[TILE_K][THREAD_N] = {0.0f};
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < TILE_K; ++j) {
                thread_tile_a[i][j] = tile_a[ty * THREAD_M + i][j];
            }
        }
#pragma unroll
        for (unsigned i = 0; i < TILE_K; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
                thread_tile_b[i][j] = tile_b[i][tx * THREAD_N + j];
            }
        }
        // 计算结果
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; j++) {
#pragma unroll
                for (unsigned tk = 0; tk < TILE_K; ++tk) {
                    ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; j++) {
            ret[(y * THREAD_M + i) * N + x * THREAD_N + j] = ret_tile[i][j];
        }
    }
}

/**
 * block内线程按顺序取global memory的float4
 *
 * 每个线程负责计算元素ret_tile[ty+i*BLOCK_M][tx+j*BLOCK_M]，而不是ret_tile[ty*THREAD_M+i][tx*THREAD_M+j]，例如
 *
 * x . x .
 *
 * . . . .
 *
 * x . x .
 *
 * . . . .
 *
 * 而不是
 *
 * x x . .
 *
 * x x . .
 *
 * . . . .
 *
 * . . . .
 *
 * @tparam TILE_M TILE尺寸，TILE_M==BLOCK_M*THREAD_M==TILE_N
 * @tparam TILE_K TILE尺寸，TILE_K*TILE_M==TILE_K*TILE_N==4*BLOCK_M*BLOCK_N
 * @tparam TILE_N TILE尺寸，TILE_N==BLOCK_N*THREAD_N==TILE_M
 * @tparam THREAD_M
 * @tparam THREAD_N
 */
template<unsigned TILE_M = 128, unsigned TILE_K = 8, unsigned TILE_N = 128,
    unsigned THREAD_M = 8, unsigned THREAD_N = 8>
__global__ void sgemm_thread_tile_v3(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    const unsigned tx = threadIdx.x, ty = threadIdx.y, tIdx = ty * blockDim.x + tx;
    const unsigned shared_a_y = tIdx * 4 / TILE_K,
            shared_a_x = tIdx * 4 % TILE_K,
            global_a_y = blockIdx.y * TILE_M + shared_a_y,
            shared_b_y = tIdx * 4 / TILE_N,
            shared_b_x = tIdx * 4 % TILE_N,
            global_b_x = blockIdx.x * TILE_N + shared_b_x;
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    for (unsigned k = 0; k < K; k += TILE_K) {
        // 填充共享内存，每个线程一个float4
sge        FLOAT4(tile_a[shared_a_y][shared_a_x]) = FLOAT4(_2D_2_1D(a, global_a_y, k + shared_a_x, K));
        FLOAT4(tile_b[shared_b_y][shared_b_x]) = FLOAT4(_2D_2_1D(b, k + shared_b_y, global_b_x, N));
        __syncthreads();
        // 填充寄存器
        float thread_tile_a[THREAD_M][TILE_K] = {0.0f}, thread_tile_b[TILE_K][THREAD_N] = {0.0f};
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < TILE_K; ++j) {
                thread_tile_a[i][j] = tile_a[ty + i * blockDim.y][j];
            }
        }
#pragma unroll
        for (unsigned i = 0; i < TILE_K; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
                thread_tile_b[i][j] = tile_b[i][tx + j * blockDim.x];
            }
        }
        // 计算结果
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; j++) {
#pragma unroll
                for (unsigned tk = 0; tk < TILE_K; ++tk) {
                    ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; j++) {
            _2D_2_1D(ret,
                     blockIdx.y * TILE_M + ty + i * blockDim.y,
                     blockIdx.x * TILE_N + tx + j * blockDim.x,
                     N) = ret_tile[i][j];
        }
    }
}

/**
 * 在v3基础上
 *
 * tile_a在shared memory转置排列，否则第一轮所有线程都读tile_a第一列，有bank conflict
 *
 * @tparam TILE_M TILE尺寸，TILE_M==BLOCK_M*THREAD_M==TILE_N
 * @tparam TILE_K TILE尺寸，TILE_K*TILE_M==TILE_K*TILE_N==4*BLOCK_M*BLOCK_N
 * @tparam TILE_N TILE尺寸，TILE_N==BLOCK_N*THREAD_N==TILE_M
 * @tparam THREAD_M
 * @tparam THREAD_N
 */
template<unsigned TILE_M = 128, unsigned TILE_K = 8, unsigned TILE_N = 128,
    unsigned THREAD_M = 8, unsigned THREAD_N = 8>
__global__ void sgemm_thread_tile_v4(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tile_a[TILE_K][TILE_M], tile_b[TILE_K][TILE_N];
    const unsigned tx = threadIdx.x, ty = threadIdx.y, tIdx = ty * blockDim.x + tx,
            shared_a_y = tIdx * 4 / TILE_K,
            shared_a_x = tIdx * 4 % TILE_K,
            global_a_y = blockIdx.y * TILE_M + shared_a_y,
            shared_b_y = tIdx * 4 / TILE_N,
            shared_b_x = tIdx * 4 % TILE_N,
            global_b_x = blockIdx.x * TILE_N + shared_b_x;
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    for (unsigned k = 0; k < K; k += TILE_K) {
        // 填充共享内存，每个线程一个float4
        // FLOAT4(tile_a[shared_a_y][shared_a_x]) = FLOAT4(_2D_2_1D(a, global_a_y, k+shared_a_x, K));
        // 寄存，转置后逐个写入shared memory
        float4 tmp_a = FLOAT4(_2D_2_1D(a, global_a_y, k + shared_a_x, K));
        tile_a[shared_a_x][shared_a_y] = tmp_a.x;
        tile_a[shared_a_x + 1][shared_a_y] = tmp_a.y;
        tile_a[shared_a_x + 2][shared_a_y] = tmp_a.z;
        tile_a[shared_a_x + 3][shared_a_y] = tmp_a.w;
        FLOAT4(tile_b[shared_b_y][shared_b_x]) = FLOAT4(_2D_2_1D(b, k + shared_b_y, global_b_x, N));
        __syncthreads();
        // 填充寄存器
        float thread_tile_a[THREAD_M][TILE_K] = {0.0f}, thread_tile_b[TILE_K][THREAD_N] = {0.0f};
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < TILE_K; ++j) {
                // thread_tile_a[i][j] = tile_a[ty + i * blockDim.y][j];
                thread_tile_a[i][j] = tile_a[j][ty + i * blockDim.y]; //转置
            }
        }
#pragma unroll
        for (unsigned i = 0; i < TILE_K; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
                thread_tile_b[i][j] = tile_b[i][tx + j * blockDim.x];
            }
        }
        // 计算结果
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; j++) {
#pragma unroll
                for (unsigned tk = 0; tk < TILE_K; ++tk) {
                    ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; j++) {
            _2D_2_1D(ret,
                     blockIdx.y * TILE_M + ty + i * blockDim.y,
                     blockIdx.x * TILE_N + tx + j * blockDim.x,
                     N) = ret_tile[i][j];
        }
    }
}

/**
 * 在v4基础上，
 *
 * 使用双缓冲，隔离每轮循环load的数据和用于计算的数据，减少synchronize
 *
 * @tparam TILE_M TILE尺寸，TILE_M==BLOCK_M*THREAD_M==TILE_N
 * @tparam TILE_K TILE尺寸，TILE_K*TILE_M==TILE_K*TILE_N==4*BLOCK_M*BLOCK_N
 * @tparam TILE_N TILE尺寸，TILE_N==BLOCK_N*THREAD_N==TILE_M
 * @tparam THREAD_M
 * @tparam THREAD_N
 */
template<unsigned TILE_M = 128, unsigned TILE_K = 8, unsigned TILE_N = 128,
    unsigned THREAD_M = 8, unsigned THREAD_N = 8>
__global__ void sgemm_thread_tile_v5(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tile_a[2][TILE_K][TILE_M], tile_b[2][TILE_K][TILE_N];
    const unsigned tx = threadIdx.x, ty = threadIdx.y, tIdx = ty * blockDim.x + tx,
            shared_a_y = tIdx * 4 / TILE_K,
            shared_a_x = tIdx * 4 % TILE_K,
            global_a_y = blockIdx.y * TILE_M + shared_a_y,
            shared_b_y = tIdx * 4 / TILE_N,
            shared_b_x = tIdx * 4 % TILE_N,
            global_b_x = blockIdx.x * TILE_N + shared_b_x;
    // 首次读取
    float4 tmp_a = FLOAT4(_2D_2_1D(a, global_a_y, shared_a_x, K));
    tile_a[0][shared_a_x][shared_a_y] = tmp_a.x;
    tile_a[0][shared_a_x + 1][shared_a_y] = tmp_a.y;
    tile_a[0][shared_a_x + 2][shared_a_y] = tmp_a.z;
    tile_a[0][shared_a_x + 3][shared_a_y] = tmp_a.w;
    FLOAT4(tile_b[0][shared_b_y][shared_b_x]) = FLOAT4(_2D_2_1D(b, shared_b_y, global_b_x, N));
    __syncthreads();
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    float thread_tile_a[THREAD_M][TILE_K] = {0.0f}, thread_tile_b[TILE_K][THREAD_N] = {0.0f};
    bool mark = false;
    for (unsigned k = TILE_K; k < K; k += TILE_K) {
        // 填充寄存器
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < TILE_K; ++j) {
                // thread_tile_a[i][j] = tile_a[ty + i * blockDim.y][j];
                thread_tile_a[i][j] = tile_a[mark][j][ty + i * blockDim.y]; //转置
            }
        }
#pragma unroll
        for (unsigned i = 0; i < TILE_K; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
                thread_tile_b[i][j] = tile_b[mark][i][tx + j * blockDim.x];
            }
        }
        // 计算结果
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
#pragma unroll
                for (unsigned tk = 0; tk < TILE_K; ++tk) {
                    ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
                }
            }
        }
        mark = !mark;
        // 填充共享内存，每个线程一个float4
        // FLOAT4(tile_a[shared_a_y][shared_a_x]) = FLOAT4(_2D_2_1D(a, global_a_y, k+shared_a_x, K));
        // 寄存，转置后逐个写入shared memory
        tmp_a = FLOAT4(_2D_2_1D(a, global_a_y, k + shared_a_x, K));
        tile_a[mark][shared_a_x][shared_a_y] = tmp_a.x;
        tile_a[mark][shared_a_x + 1][shared_a_y] = tmp_a.y;
        tile_a[mark][shared_a_x + 2][shared_a_y] = tmp_a.z;
        tile_a[mark][shared_a_x + 3][shared_a_y] = tmp_a.w;
        FLOAT4(tile_b[mark][shared_b_y][shared_b_x]) = FLOAT4(_2D_2_1D(b, k + shared_b_y, global_b_x, N));
        __syncthreads();
    }
    // 使用最后一次读取的数据
    // 填充寄存器
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < TILE_K; ++j) {
            // thread_tile_a[i][j] = tile_a[ty + i * blockDim.y][j];
            thread_tile_a[i][j] = tile_a[mark][j][ty + i * blockDim.y]; //转置
        }
    }
#pragma unroll
    for (unsigned i = 0; i < TILE_K; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; ++j) {
            thread_tile_b[i][j] = tile_b[mark][i][tx + j * blockDim.x];
        }
    }
    // 计算结果
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; ++j) {
#pragma unroll
            for (unsigned tk = 0; tk < TILE_K; ++tk) {
                ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
            }
        }
    }
    // 写回结果
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; ++j) {
            _2D_2_1D(ret,
                     blockIdx.y * TILE_M + ty + i * blockDim.y,
                     blockIdx.x * TILE_N + tx + j * blockDim.x,
                     N) = ret_tile[i][j];
        }
    }
}


/**
 * 在v5基础上，
 *
 * 异步搬运全局内存数据到共享内存，实现真正的compute和load同时
 *
 * @tparam TILE_M TILE尺寸，TILE_M==BLOCK_M*THREAD_M==TILE_N
 * @tparam TILE_K TILE尺寸
 * @tparam TILE_N TILE尺寸，TILE_N==BLOCK_N*THREAD_N==TILE_M
 * @tparam THREAD_M
 * @tparam THREAD_N
 */
template<unsigned TILE_M = 128, unsigned TILE_K = 8, unsigned TILE_N = 128,
    unsigned THREAD_M = 8, unsigned THREAD_N = 8>
__global__ void sgemm_thread_tile_v6(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tile_a[2][TILE_K][TILE_M], tile_b[2][TILE_K][TILE_N];
    const unsigned tx = threadIdx.x, ty = threadIdx.y, tIdx = ty * blockDim.x + tx,
            strip = blockDim.x * blockDim.y;
    // async load firstly
#pragma unroll
    for (unsigned idx = tIdx; idx < TILE_M * TILE_K; idx += strip) {
        const unsigned shared_a_y = idx / TILE_K, shared_a_x = idx % TILE_K,
                global_a_y = blockIdx.y * TILE_M + shared_a_y;
        // 转置
        __pipeline_memcpy_async(&tile_a[0][shared_a_x][shared_a_y],
                                &_2D_2_1D(a, global_a_y, shared_a_x, K),
                                sizeof(float));
    }
#pragma unroll
    for (unsigned idx = tIdx; idx < TILE_K * TILE_N; idx += strip) {
        const unsigned shared_b_y = idx / TILE_N, shared_b_x = idx % TILE_N,
                global_b_x = blockIdx.x * TILE_N + shared_b_x;
        __pipeline_memcpy_async(&tile_b[0][shared_b_y][shared_b_x],
                                &_2D_2_1D(b, shared_b_y, global_b_x, N),
                                sizeof(float));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    float thread_tile_a[THREAD_M][TILE_K] = {0.0f}, thread_tile_b[TILE_K][THREAD_N] = {0.0f};
    bool mark = false;
    for (unsigned k = TILE_K; k < K; k += TILE_K, mark = !mark) {
        // async load
#pragma unroll
        for (unsigned idx = tIdx; idx < TILE_M * TILE_K; idx += strip) {
            const unsigned shared_a_y = idx / TILE_K, shared_a_x = idx % TILE_K,
                    global_a_y = blockIdx.y * TILE_M + shared_a_y;
            // 转置
            __pipeline_memcpy_async(&tile_a[!mark][shared_a_x][shared_a_y],
                                    &_2D_2_1D(a, global_a_y, k + shared_a_x, K),
                                    sizeof(float));
        }
#pragma unroll
        for (unsigned idx = tIdx; idx < TILE_K * TILE_N; idx += strip) {
            const unsigned shared_b_y = idx / TILE_N, shared_b_x = idx % TILE_N,
                    global_b_x = blockIdx.x * TILE_N + shared_b_x;
            __pipeline_memcpy_async(&tile_b[!mark][shared_b_y][shared_b_x],
                                    &_2D_2_1D(b, k + shared_b_y, global_b_x, N),
                                    sizeof(float));
        }
        __pipeline_commit();
        // 填充寄存器
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < TILE_K; ++j) {
                // thread_tile_a[i][j] = tile_a[ty + i * blockDim.y][j];
                thread_tile_a[i][j] = tile_a[mark][j][ty + i * blockDim.y]; //转置
            }
        }
#pragma unroll
        for (unsigned i = 0; i < TILE_K; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
                thread_tile_b[i][j] = tile_b[mark][i][tx + j * blockDim.x];
            }
        }
        // 计算结果
#pragma unroll
        for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_N; ++j) {
#pragma unroll
                for (unsigned tk = 0; tk < TILE_K; ++tk) {
                    ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
                }
            }
        }
        __pipeline_wait_prior(0);
        __syncthreads();
    }
    // 使用最后一次读取的数据
    // 填充寄存器
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < TILE_K; ++j) {
            // thread_tile_a[i][j] = tile_a[ty + i * blockDim.y][j];
            thread_tile_a[i][j] = tile_a[mark][j][ty + i * blockDim.y]; //转置
        }
    }
#pragma unroll
    for (unsigned i = 0; i < TILE_K; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; ++j) {
            thread_tile_b[i][j] = tile_b[mark][i][tx + j * blockDim.x];
        }
    }
    // 计算结果
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; ++j) {
#pragma unroll
            for (unsigned tk = 0; tk < TILE_K; ++tk) {
                ret_tile[i][j] += thread_tile_a[i][tk] * thread_tile_b[tk][j];
            }
        }
    }
    // 写回结果
#pragma unroll
    for (unsigned i = 0; i < THREAD_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_N; ++j) {
            _2D_2_1D(ret,
                     blockIdx.y * TILE_M + ty + i * blockDim.y,
                     blockIdx.x * TILE_N + tx + j * blockDim.x,
                     N) = ret_tile[i][j];
        }
    }
}

/**
 * 算法重构，使用外积思想
 *
 * 单个线程 数据缓存 寄存器占用 THREAD_TILE_M*TILE_K+THREAD_TILE_N*TILE_K -> THREAD_TILE_M+THREAD_TILE_M
 *
 *
 * @tparam TILE_M TILE尺寸，TILE_M==BLOCK_M*THREAD_M==TILE_N
 * @tparam TILE_K TILE尺寸，TILE_K*TILE_M==TILE_K*TILE_N==4*BLOCK_M*BLOCK_N
 * @tparam TILE_N TILE尺寸，TILE_N==BLOCK_N*THREAD_N==TILE_M
 * @tparam THREAD_TILE_M
 * @tparam THREAD_TILE_N
 */
template<unsigned TILE_M = 64, unsigned TILE_K = 16, unsigned TILE_N = 64,
    unsigned THREAD_TILE_M = 4, unsigned THREAD_TILE_N = 4>
__global__ void sgemm_thread_tile_v7(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    __shared__ float tile_a[2][TILE_K][TILE_M], //padding
            tile_b[2][TILE_K][TILE_N];
    const unsigned threadIdxInBlock = threadIdx.y * blockDim.x + threadIdx.x,
            NUM_THREAD_PER_BLOCK = blockDim.x * blockDim.y;
    const unsigned TILE_i = blockIdx.y, TILE_j = blockIdx.x,
            TILE_OFFSET_i = TILE_i * TILE_M, TILE_OFFSET_j = TILE_j * TILE_N;
    // first load
    bool flag = false;
    for (unsigned i = threadIdxInBlock * 4; i < TILE_M * TILE_K; i += NUM_THREAD_PER_BLOCK * 4) {
        const unsigned shared_i = i / TILE_K, shared_j = i % TILE_K;
        float4 tmp = FLOAT4(_2D_2_1D(a, TILE_OFFSET_i + shared_i, shared_j, K));
        // 转置
        tile_a[flag][shared_j + 0][shared_i] = tmp.x;
        tile_a[flag][shared_j + 1][shared_i] = tmp.y;
        tile_a[flag][shared_j + 2][shared_i] = tmp.z;
        tile_a[flag][shared_j + 3][shared_i] = tmp.w;
    }
    for (unsigned i = threadIdxInBlock * 4; i < TILE_N * TILE_K; i += NUM_THREAD_PER_BLOCK * 4) {
        const unsigned shared_i = i / TILE_N, shared_j = i % TILE_N;
        FLOAT4(tile_b[flag][shared_i][shared_j]) = FLOAT4(_2D_2_1D(b, shared_i, TILE_OFFSET_j + shared_j, N));
    }
    __syncthreads();
    const unsigned THREAD_TILE_i = threadIdx.y, THREAD_TILE_j = threadIdx.x,
            THREAD_TILE_OFFSET_i = THREAD_TILE_i * THREAD_TILE_M, THREAD_TILE_OFFSET_j = THREAD_TILE_j * THREAD_TILE_N;
    float thread_tile[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};
    float thread_tile_a[THREAD_TILE_M], thread_tile_b[THREAD_TILE_N];
#pragma unroll
    for (unsigned k = TILE_K; k < K; k += TILE_K) {
        // compute
#pragma unroll
        for (unsigned tk = 0; tk < TILE_K; ++tk) {
#pragma unroll
            for (unsigned i = 0; i < THREAD_TILE_M; i += 4) {
                FLOAT4(thread_tile_a[i]) = FLOAT4(tile_a[flag][tk][THREAD_TILE_OFFSET_i + i]);
            }
#pragma unroll
            for (unsigned i = 0; i < THREAD_TILE_N; i += 4) {
                FLOAT4(thread_tile_b[i]) = FLOAT4(tile_b[flag][tk][THREAD_TILE_OFFSET_j + i]);
            }
#pragma unroll
            for (unsigned i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
                for (unsigned j = 0; j < THREAD_TILE_N; ++j) {
                    thread_tile[i][j] += thread_tile_a[i] * thread_tile_b[j];
                }
            }
        }
        // load
        flag = !flag;
        for (unsigned i = threadIdxInBlock * 4; i < TILE_M * TILE_K; i += NUM_THREAD_PER_BLOCK * 4) {
            const unsigned shared_i = i / TILE_K, shared_j = i % TILE_K;
            float4 tmp = FLOAT4(_2D_2_1D(a, TILE_OFFSET_i + shared_i, k + shared_j, K));
            // 转置
            tile_a[flag][shared_j + 0][shared_i] = tmp.x;
            tile_a[flag][shared_j + 1][shared_i] = tmp.y;
            tile_a[flag][shared_j + 2][shared_i] = tmp.z;
            tile_a[flag][shared_j + 3][shared_i] = tmp.w;
        }
        for (unsigned i = threadIdxInBlock * 4; i < TILE_N * TILE_K; i += NUM_THREAD_PER_BLOCK * 4) {
            const unsigned shared_i = i / TILE_N, shared_j = i % TILE_N;
            FLOAT4(tile_b[flag][shared_i][shared_j]) = FLOAT4(_2D_2_1D(b, k + shared_i, TILE_OFFSET_j + shared_j, N));
        }
        // sync
        __syncthreads();
    }
    // last compute
#pragma unroll
    for (unsigned tk = 0; tk < TILE_K; ++tk) {
#pragma unroll
        for (unsigned i = 0; i < THREAD_TILE_M; i += 4) {
            FLOAT4(thread_tile_a[i]) = FLOAT4(tile_a[flag][tk][THREAD_TILE_OFFSET_i + i]);
        }
#pragma unroll
        for (unsigned i = 0; i < THREAD_TILE_N; i += 4) {
            FLOAT4(thread_tile_b[i]) = FLOAT4(tile_b[flag][tk][THREAD_TILE_OFFSET_j + i]);
        }
#pragma unroll
        for (unsigned i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
            for (unsigned j = 0; j < THREAD_TILE_N; ++j) {
                thread_tile[i][j] += thread_tile_a[i] * thread_tile_b[j];
            }
        }
    }
    // store
#pragma unroll
    for (unsigned i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
        for (unsigned j = 0; j < THREAD_TILE_N; j += 4) {
            FLOAT4(_2D_2_1D(ret, TILE_OFFSET_i + THREAD_TILE_OFFSET_i + i, TILE_OFFSET_j + THREAD_TILE_OFFSET_j + j, N))
                    =
                    FLOAT4(thread_tile[i][j]);
        }
    }
}


/**
 * 一维warp，每个warp处理16*16 tile
 *
 * 读入shared转为half
 *
 * @param K
 * @param N
 * @param a
 * @param b
 * @param ret
 */
__global__ void sgemm_tensor_core_v0(unsigned K, unsigned N, const float *a, const float *b, float *ret);

template<unsigned TILE_M, unsigned TILE_N>
__global__ void sgemm_tensor_core_v1(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    constexpr unsigned WMMA_TILE_M = 16, WMMA_TILE_K = 16, WMMA_TILE_N = 16;
    constexpr unsigned WMMA_TILE_IDX_STRIDE = TILE_N / WMMA_TILE_N;
    using namespace nvcuda;
    constexpr unsigned TILE_K = WMMA_TILE_K;
    // 双缓冲
    __shared__ half tile_a[2][TILE_M][TILE_K], tile_b[2][TILE_K][TILE_N];
    // block内序号
    const unsigned threadIdxInBlock = threadIdx.x, NUM_THREAD_IN_BLOCK = blockDim.x, warpIdx =
            threadIdxInBlock / WARP_SIZE;
    const unsigned TILE_IDX_i = blockIdx.y, TILE_IDX_j = blockIdx.x;
    const unsigned WMMA_TILE_IDX_i = warpIdx / WMMA_TILE_IDX_STRIDE, WMMA_TILE_IDX_j = warpIdx % WMMA_TILE_IDX_STRIDE;
    // first load to shared
    bool flag = false;
#pragma unroll
    for (unsigned i = threadIdxInBlock; i < TILE_M * TILE_K; i += NUM_THREAD_IN_BLOCK) {
        const unsigned shared_a_i = i / TILE_K, shared_a_j = i % TILE_K;
        tile_a[flag][shared_a_i][shared_a_j] = __float2half(
            _2D_2_1D(a, TILE_IDX_i * TILE_M + shared_a_i, shared_a_j, K));
    }
#pragma unroll
    for (unsigned i = threadIdxInBlock; i < TILE_N * TILE_K; i += NUM_THREAD_IN_BLOCK) {
        const unsigned shared_b_i = i / TILE_N, shared_b_j = i % TILE_N;
        tile_b[flag][shared_b_i][shared_b_j] = __float2half(
            _2D_2_1D(b, shared_b_i, TILE_IDX_j * TILE_N + shared_b_j, N));
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);
#pragma unroll
    for (unsigned k = WMMA_TILE_K; k < K; k += WMMA_TILE_K) {
        wmma::load_matrix_sync(frag_a, &tile_a[flag][WMMA_TILE_IDX_i * WMMA_TILE_M][0], TILE_K);
        wmma::load_matrix_sync(frag_b, &tile_b[flag][0][WMMA_TILE_IDX_j * WMMA_TILE_N], TILE_N);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        flag = !flag;
#pragma unroll
        for (unsigned i = threadIdxInBlock; i < TILE_M * TILE_K; i += NUM_THREAD_IN_BLOCK) {
            const unsigned shared_a_i = i / TILE_K, shared_a_j = i % TILE_K;
            tile_a[flag][shared_a_i][shared_a_j] = __float2half(
                _2D_2_1D(a, TILE_IDX_i * TILE_M + shared_a_i, k + shared_a_j, K));
        }
#pragma unroll
        for (unsigned i = threadIdxInBlock; i < TILE_N * TILE_K; i += NUM_THREAD_IN_BLOCK) {
            const unsigned shared_b_i = i / TILE_N, shared_b_j = i % TILE_N;
            tile_b[flag][shared_b_i][shared_b_j] = __float2half(
                _2D_2_1D(b, k + shared_b_i, TILE_IDX_j * TILE_N + shared_b_j, N));
        }
        __syncthreads();
    }
    // last compute
    wmma::load_matrix_sync(frag_a, &tile_a[flag][WMMA_TILE_IDX_i * WMMA_TILE_M][0], TILE_K);
    wmma::load_matrix_sync(frag_b, &tile_b[flag][0][WMMA_TILE_IDX_j * WMMA_TILE_N], TILE_N);
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    // load to global
    const unsigned ret_i = blockIdx.y * TILE_M + WMMA_TILE_IDX_i * WMMA_TILE_M;
    const unsigned ret_j = blockIdx.x * TILE_N + WMMA_TILE_IDX_j * WMMA_TILE_N;
    wmma::store_matrix_sync(&_2D_2_1D(ret, ret_i, ret_j, N), frag_c, N, wmma::mem_row_major);
}

/**
 * 在v1基础上，tensor core输入类型改为float
 */
template<unsigned TILE_M, unsigned TILE_N>
__global__ void sgemm_tensor_core_v2(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    constexpr unsigned WMMA_TILE_M = 16, WMMA_TILE_K = 8, WMMA_TILE_N = 16;
    constexpr unsigned WMMA_TILE_IDX_STRIDE = TILE_N / WMMA_TILE_N;
    using namespace nvcuda;
    constexpr unsigned TILE_K = WMMA_TILE_K;
    // 双缓冲
    __shared__ float tile_a[2][TILE_M][TILE_K], tile_b[2][TILE_K][TILE_N];
    // block内序号
    const unsigned threadIdxInBlock = threadIdx.x, NUM_THREAD_IN_BLOCK = blockDim.x, warpIdx =
            threadIdxInBlock / WARP_SIZE;
    const unsigned TILE_IDX_i = blockIdx.y, TILE_IDX_j = blockIdx.x;
    const unsigned WMMA_TILE_IDX_i = warpIdx / WMMA_TILE_IDX_STRIDE, WMMA_TILE_IDX_j = warpIdx % WMMA_TILE_IDX_STRIDE;
    // first load to shared
    bool flag = false;
#pragma unroll
    for (unsigned i = threadIdxInBlock; i < TILE_M * TILE_K; i += NUM_THREAD_IN_BLOCK) {
        const unsigned shared_a_i = i / TILE_K, shared_a_j = i % TILE_K;
        tile_a[flag][shared_a_i][shared_a_j] = wmma::__float_to_tf32(
            _2D_2_1D(a, TILE_IDX_i * TILE_M + shared_a_i, shared_a_j, K));
    }
#pragma unroll
    for (unsigned i = threadIdxInBlock; i < TILE_N * TILE_K; i += NUM_THREAD_IN_BLOCK) {
        const unsigned shared_b_i = i / TILE_N, shared_b_j = i % TILE_N;
        tile_b[flag][shared_b_i][shared_b_j] = wmma::__float_to_tf32(
            _2D_2_1D(b, shared_b_i, TILE_IDX_j * TILE_N + shared_b_j, N));
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, wmma::precision::tf32, wmma::row_major>
            frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, wmma::precision::tf32, wmma::row_major>
            frag_b;
    wmma::fragment<wmma::accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);
#pragma unroll
    for (unsigned k = WMMA_TILE_K; k < K; k += WMMA_TILE_K) {
        wmma::load_matrix_sync(frag_a, &tile_a[flag][WMMA_TILE_IDX_i * WMMA_TILE_M][0], TILE_K);
        wmma::load_matrix_sync(frag_b, &tile_b[flag][0][WMMA_TILE_IDX_j * WMMA_TILE_N], TILE_N);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
        flag = !flag;
#pragma unroll
        for (unsigned i = threadIdxInBlock; i < TILE_M * TILE_K; i += NUM_THREAD_IN_BLOCK) {
            const unsigned shared_a_i = i / TILE_K, shared_a_j = i % TILE_K;
            tile_a[flag][shared_a_i][shared_a_j] = wmma::__float_to_tf32(
                _2D_2_1D(a, TILE_IDX_i * TILE_M + shared_a_i, k + shared_a_j, K));
        }
#pragma unroll
        for (unsigned i = threadIdxInBlock; i < TILE_N * TILE_K; i += NUM_THREAD_IN_BLOCK) {
            const unsigned shared_b_i = i / TILE_N, shared_b_j = i % TILE_N;
            tile_b[flag][shared_b_i][shared_b_j] = wmma::__float_to_tf32(
                _2D_2_1D(b, k + shared_b_i, TILE_IDX_j * TILE_N + shared_b_j, N));
        }
        __syncthreads();
    }
    // last compute
    wmma::load_matrix_sync(frag_a, &tile_a[flag][WMMA_TILE_IDX_i * WMMA_TILE_M][0], TILE_K);
    wmma::load_matrix_sync(frag_b, &tile_b[flag][0][WMMA_TILE_IDX_j * WMMA_TILE_N], TILE_N);
    wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    // load to global
    const unsigned ret_i = blockIdx.y * TILE_M + WMMA_TILE_IDX_i * WMMA_TILE_M;
    const unsigned ret_j = blockIdx.x * TILE_N + WMMA_TILE_IDX_j * WMMA_TILE_N;
    wmma::store_matrix_sync(&_2D_2_1D(ret, ret_i, ret_j, N), frag_c, N, wmma::mem_row_major);
}

#endif // CUDA_TUTORIAL_EXAMPLE_KERNEL_H
