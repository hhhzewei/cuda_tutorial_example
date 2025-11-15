//
// Created by hzw on 2025/10/25.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_KERNEL_H

#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FLOAT4(x) (*((float4*)(&x)))
#define TWO_D_2_ONE_D(a,i,j,step) (a)[(i)*(step)+(j)]

// add
__global__ void add(unsigned n, float *a, float *b, float *ret);

/**
 *
 * @param N % 4 == 0
 * @param a
 * @param b
 * @param ret
 */
__global__ void add_float4(unsigned N, float *a, float *b, float *ret);

// dot
__global__ void dot(unsigned N, float *a, float *b, float *ret);

template<unsigned WARP_SIZE>
__device__ __forceinline__ void reduce_add_v0(float &value) {
#pragma unroll
    for (unsigned offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
}

template<unsigned WARP_SIZE>
__device__ __forceinline__ void reduce_add_v1(float &value) {
#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffff, value, offset);
    }
}

template<const int BLOCK_DIM>
__global__ void dot_share(unsigned N, float *a, float *b, float *ret) {
    __shared__ float tmp[BLOCK_DIM];
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    tmp[tid] = 0;
    unsigned int strip = gridDim.x * blockDim.x;
    for (unsigned int i = idx; i < N; i += strip) {
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

__global__ void dot_shared_external(unsigned N, float *a, float *b, float *ret);

/**
 *
 * warp_num = blockDim / warpSize <=  * warpSize
 */
template<unsigned WARP_NUM, unsigned WARP_SIZE = 32>
__global__ void dot_warp_shuffle_v0(unsigned N, float *a, float *b, float *ret) {
    __shared__ float tmp[WARP_NUM];
    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x, strip = gridDim.x * blockDim.x;
    unsigned warpNum = CEIL(blockDim.x, warpSize), warpIdx = threadIdx.x / warpSize, laneIdx =
            threadIdx.x - warpIdx * warpSize;
    float value = 0.0f;
    for (unsigned i = idx; i < N; i += strip) {
        value += a[i] * b[i];
    }
    __syncwarp();
    reduce_add_v0<WARP_SIZE>(value);
    if (laneIdx == 0) {
        tmp[warpIdx] = value;
    }
    __syncthreads();
    if (warpIdx == 0) {
        value = laneIdx < warpNum ? tmp[laneIdx] : 0.0f;
        reduce_add_v0<WARP_SIZE>(value);
        if (laneIdx == 0) {
            atomicAdd(ret, value);
        }
    }
}

template<unsigned WARP_NUM, unsigned WARP_SIZE = 32>
__global__ void dot_warp_shuffle_v1(unsigned N, float *a, float *b, float *ret) {
    __shared__ float tmp[WARP_NUM];
    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x, strip = gridDim.x * blockDim.x;
    unsigned warpNum = CEIL(blockDim.x, warpSize), warpIdx = threadIdx.x / warpSize, laneIdx =
            threadIdx.x - warpIdx * warpSize;
    float value = 0.0f;
    for (unsigned i = idx; i < N; i += strip) {
        value += a[i] * b[i];
    }
    __syncwarp();
    reduce_add_v1<WARP_SIZE>(value);
    if (laneIdx == 0) {
        tmp[warpIdx] = value;
    }
    __syncthreads();
    if (warpIdx == 0) {
        value = laneIdx < warpNum ? tmp[laneIdx] : 0.0f;
        reduce_add_v1<WARP_SIZE>(value);
        if (laneIdx == 0) {
            atomicAdd(ret, value);
        }
    }
}


// transport
__global__ void transpose_naive(unsigned M, unsigned N, float *input, float *output);

template<unsigned WARP_SIZE>
__global__ void transpose_shared(unsigned M, unsigned N, float *input, float *output) {
    // padding
    __shared__ float tile[WARP_SIZE][WARP_SIZE];
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x,
            y = blockIdx.y * blockDim.y + threadIdx.y,
            tx = threadIdx.x,
            ty = threadIdx.y;
    if (x < N && y < M) {
        tile[ty][tx] = input[y * N + x];
    }
    __syncthreads();
    unsigned x1 = blockDim.x * blockIdx.x + ty, y1 = blockDim.y * blockIdx.y + tx;
    if (x1 < N && y1 < M) {
        output[x1 * M + y1] = tile[tx][ty];
    }
}

template<unsigned WARP_SIZE>
__global__ void transpose_padding(unsigned M, unsigned N, float *input, float *output) {
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
    unsigned x1 = blockDim.x * blockIdx.x + ty, y1 = blockDim.y * blockIdx.y + tx;
    if (x1 < N && y1 < M) {
        output[x1 * M + y1] = tile[tx][ty];
    }
}

template<unsigned WARP_SIZE>
__global__ void transpose_swizzle(unsigned M, unsigned N, float *input, float *output) {
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
    unsigned x1 = blockDim.x * blockIdx.x + ty, y1 = blockDim.y * blockIdx.y + tx;
    if (x1 < N && y1 < M) {
        output[x1 * M + y1] = tile[tx][ty ^ tx];
    }
}

// sgemm
__global__ void sgemm_naive(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret);

template<unsigned TILE_M, unsigned TILE_N, unsigned TILE_K>
__global__ void sgemm_block_tile(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    __shared__ float tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    unsigned tx = threadIdx.x, ty = threadIdx.y,
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
__global__ void sgemm_thread_tile_v0(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    __shared__ float tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    unsigned tx = threadIdx.x * THREAD_N, ty = threadIdx.y * THREAD_M,
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
 * 同warp线程访问b矩阵同一行会bank conflict，如THREAD_N=2时，访问bank序列为0,2,...,30,0,2,...,30
 *
 * 添加thread tile内x方向的offset=tx/warp_circle
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
__global__ void sgemm_thread_tile_v1(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    constexpr unsigned TILE_M = BLOCK_M * THREAD_M, TILE_N = BLOCK_N * THREAD_N;
    __shared__ float tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    unsigned tx = threadIdx.x, ty = threadIdx.y,
            x = blockDim.x * blockIdx.x + threadIdx.x, y = blockDim.y * blockIdx.y + threadIdx.y;
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    // unsigned warp_circle = CEIL(warpSize, THREAD_N);
    // offset=tx/warp_circle
    unsigned offset = tx >> WARP_CIRCLE_LOG;
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
__global__ void sgemm_thread_tile_v2(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    // padding
    // 一个线程x方向负责THREAD_N个元素，读共享内存会bank_conflict
    __shared__ float tile_a[TILE_M][TILE_K + 1], tile_b[TILE_K][TILE_N + 1];
    unsigned tx = threadIdx.x, ty = threadIdx.y, tIdx = ty * blockDim.x + tx,
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
 * 每个线程负责元素ret_tile[ty+i*BLOCK_M][tx+j*BLOCK_M]，而不是ret_tile[ty*THREAD_M+i][tx*THREAD_M+j]，例如
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
__global__ void sgemm_thread_tile_v3(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    __shared__ float tile_a[TILE_M][TILE_K], tile_b[TILE_K][TILE_N];
    unsigned tx = threadIdx.x, ty = threadIdx.y, tIdx = ty * blockDim.x + tx;
    unsigned shared_a_y = tIdx * 4 / TILE_K,
            shared_a_x = tIdx * 4 % TILE_K,
            global_a_y = blockIdx.y * TILE_M + shared_a_y,
            shared_b_y = tIdx * 4 / TILE_N,
            shared_b_x = tIdx * 4 % TILE_N,
            global_b_x = blockIdx.x * TILE_N + shared_b_x;
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    for (unsigned k = 0; k < K; k += TILE_K) {
        // 填充共享内存，每个线程一个float4
        FLOAT4(tile_a[shared_a_y][shared_a_x]) = FLOAT4(TWO_D_2_ONE_D(a, global_a_y, k+shared_a_x, K));
        FLOAT4(tile_b[shared_b_y][shared_b_x]) = FLOAT4(TWO_D_2_ONE_D(b, k+shared_b_y, global_b_x, N));
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
            TWO_D_2_ONE_D(ret,
                          blockIdx.y * TILE_M + ty + i * blockDim.y,
                          blockIdx.x * TILE_N + tx + j * blockDim.x,
                          N) = ret_tile[i][j];
        }
    }
}

/**
 * block内线程按顺序取global memory的float4
 *
 * 每个线程负责元素ret_tile[ty+i*BLOCK_M][tx+j*BLOCK_M]，而不是ret_tile[ty*THREAD_M+i][tx*THREAD_M+j]，例如
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
__global__ void sgemm_thread_tile_v4(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    __shared__ float tile_a[TILE_K][TILE_M], tile_b[TILE_K][TILE_N];
    unsigned tx = threadIdx.x, ty = threadIdx.y, tIdx = ty * blockDim.x + tx;
    unsigned shared_a_y = tIdx * 4 / TILE_K,
            shared_a_x = tIdx * 4 % TILE_K,
            global_a_y = blockIdx.y * TILE_M + shared_a_y,
            shared_b_y = tIdx * 4 / TILE_N,
            shared_b_x = tIdx * 4 % TILE_N,
            global_b_x = blockIdx.x * TILE_N + shared_b_x;
    float ret_tile[THREAD_M][THREAD_N] = {0.0f};
    for (unsigned k = 0; k < K; k += TILE_K) {
        // 填充共享内存，每个线程一个float4
        // FLOAT4(tile_a[shared_a_y][shared_a_x]) = FLOAT4(TWO_D_2_ONE_D(a, global_a_y, k+shared_a_x, K));
        // 寄存，转置后逐个写入shared memory
        float4 tmp_a = FLOAT4(TWO_D_2_ONE_D(a, global_a_y, k+shared_a_x, K));
        tile_a[shared_a_x][shared_a_y] = tmp_a.x;
        tile_a[shared_a_x + 1][shared_a_y] = tmp_a.y;
        tile_a[shared_a_x + 2][shared_a_y] = tmp_a.z;
        tile_a[shared_a_x + 3][shared_a_y] = tmp_a.w;
        FLOAT4(tile_b[shared_b_y][shared_b_x]) = FLOAT4(TWO_D_2_ONE_D(b, k+shared_b_y, global_b_x, N));
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
            TWO_D_2_ONE_D(ret,
                          blockIdx.y * TILE_M + ty + i * blockDim.y,
                          blockIdx.x * TILE_N + tx + j * blockDim.x,
                          N) = ret_tile[i][j];
        }
    }
}

/**
 * block内线程按顺序取global memory的float4
 *
 * 每个线程负责元素ret_tile[ty+i*BLOCK_M][tx+j*BLOCK_M]，而不是ret_tile[ty*THREAD_M+i][tx*THREAD_M+j]，例如
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
__global__ void sgemm_thread_tile_v5(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    __shared__ float tile_a[2][TILE_K][TILE_M], tile_b[2][TILE_K][TILE_N];
    unsigned tx = threadIdx.x, ty = threadIdx.y, tIdx = ty * blockDim.x + tx;
    unsigned shared_a_y = tIdx * 4 / TILE_K,
            shared_a_x = tIdx * 4 % TILE_K,
            global_a_y = blockIdx.y * TILE_M + shared_a_y,
            shared_b_y = tIdx * 4 / TILE_N,
            shared_b_x = tIdx * 4 % TILE_N,
            global_b_x = blockIdx.x * TILE_N + shared_b_x;
    // 首次读取
    float4 tmp_a = FLOAT4(TWO_D_2_ONE_D(a, global_a_y, shared_a_x, K));
    tile_a[0][shared_a_x][shared_a_y] = tmp_a.x;
    tile_a[0][shared_a_x + 1][shared_a_y] = tmp_a.y;
    tile_a[0][shared_a_x + 2][shared_a_y] = tmp_a.z;
    tile_a[0][shared_a_x + 3][shared_a_y] = tmp_a.w;
    FLOAT4(tile_b[0][shared_b_y][shared_b_x]) = FLOAT4(TWO_D_2_ONE_D(b, shared_b_y, global_b_x, N));
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
        // FLOAT4(tile_a[shared_a_y][shared_a_x]) = FLOAT4(TWO_D_2_ONE_D(a, global_a_y, k+shared_a_x, K));
        // 寄存，转置后逐个写入shared memory
        tmp_a = FLOAT4(TWO_D_2_ONE_D(a, global_a_y, k+shared_a_x, K));
        tile_a[mark][shared_a_x][shared_a_y] = tmp_a.x;
        tile_a[mark][shared_a_x + 1][shared_a_y] = tmp_a.y;
        tile_a[mark][shared_a_x + 2][shared_a_y] = tmp_a.z;
        tile_a[mark][shared_a_x + 3][shared_a_y] = tmp_a.w;
        FLOAT4(tile_b[mark][shared_b_y][shared_b_x]) = FLOAT4(TWO_D_2_ONE_D(b, k+shared_b_y, global_b_x, N));
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
            TWO_D_2_ONE_D(ret,
                          blockIdx.y * TILE_M + ty + i * blockDim.y,
                          blockIdx.x * TILE_N + tx + j * blockDim.x,
                          N) = ret_tile[i][j];
        }
    }
}


#endif // CUDA_TUTORIAL_EXAMPLE_KERNEL_H
