#pragma once

#include <cuda/barrier>

template<typename T, unsigned kBlockM, unsigned kBlockN, unsigned kBlockK,
    typename VecType, unsigned kNumPer128b>
__device__ __forceinline__ void load(const T *a, const T *b, const unsigned N, const unsigned K,
                                     T s_a[kBlockM][kBlockK], T s_b[kBlockK][kBlockN],
                                     const unsigned block_x, const unsigned block_y,
                                     const unsigned num_thread, const unsigned thread_idx,
                                     cuda::barrier<cuda::thread_scope_block> &barrier, const unsigned block_k_start) {
    for (unsigned i = thread_idx * kNumPer128b; i < kBlockM * kBlockK; i += num_thread * kNumPer128b) {
        const unsigned s_row_idx = i / kBlockK;
        const unsigned s_col_idx = i % kBlockK;
        const unsigned g_row_idx = block_y * kBlockM + s_row_idx;
        const unsigned g_col_idx = block_k_start + s_col_idx;
        cuda::memcpy_async(&s_a[s_row_idx][s_col_idx],
                           &a[g_row_idx * K + g_col_idx],
                           cuda::aligned_size_t<alignof(VecType)>(sizeof(VecType)), barrier);
    }
    for (unsigned i = thread_idx * kNumPer128b; i < kBlockN * kBlockK; i += num_thread * kNumPer128b) {
        const unsigned s_row_idx = i / kBlockN;
        const unsigned s_col_idx = i % kBlockN;
        const unsigned g_col_idx = block_x * kBlockN + s_col_idx;
        const unsigned g_row_idx = block_k_start + s_row_idx;
        cuda::memcpy_async(&s_b[s_row_idx][s_col_idx],
                           &b[g_row_idx * N + g_col_idx],
                           cuda::aligned_size_t<alignof(VecType)>(sizeof(VecType)), barrier);
    }
}


template<typename T, unsigned kBlockM, unsigned kBlockN, unsigned kBlockK, unsigned kThreadM, unsigned kThreadN,
    unsigned kThreadLayoutM, unsigned kThreadLayoutN>
__device__ __forceinline__ void compute(T s_a[kBlockM][kBlockK], T s_b[kBlockK][kBlockN],
                                        T r_a[kThreadM], T r_b[kThreadN], T r_c[kThreadM][kThreadN],
                                        const unsigned thread_layout_i, const unsigned thread_layout_j) {
#pragma unroll
    for (unsigned k = 0; k < kBlockK; ++k) {
#pragma unroll
        for (unsigned i = 0; i < kThreadM; ++i) {
            r_a[i] = s_a[thread_layout_i + i * kThreadLayoutM][k];
        }
#pragma unroll
        for (unsigned i = 0; i < kThreadN; ++i) {
            r_b[i] = s_b[k][thread_layout_j + i * kThreadLayoutN];
        }
#pragma unroll
        for (unsigned i = 0; i < kThreadM; ++i) {
#pragma unroll
            for (unsigned j = 0; j < kThreadN; ++j) {
                r_c[i][j] += r_a[i] * r_b[j];
            }
        }
    }
}

template<
    typename T,
    unsigned kBlockM, unsigned kBlockN, unsigned kBlockK, unsigned kThreadM, unsigned kThreadN>
__global__ void gemm_sm80_cuda(const T *a, const T *b, T *c,
                               const unsigned M, const unsigned N, const unsigned K) {
    constexpr unsigned kThreadLayoutM = kBlockM / kThreadM;
    constexpr unsigned kThreadLayoutN = kBlockN / kThreadN;
    using VecType = float4;
    // using VecTYpe = float;
    constexpr unsigned kNumPerVec = sizeof(VecType) / sizeof(T);
    __align__(alignof(VecType)) __shared__ T s_a[2][kBlockM][kBlockK];
    __align__(alignof(VecType)) __shared__ T s_b[2][kBlockK][kBlockN];
    const unsigned block_x = blockIdx.x;
    const unsigned block_y = blockIdx.y;
    const unsigned num_thread = blockDim.x * blockDim.y;
    const unsigned thread_idx = threadIdx.x + threadIdx.y * blockDim.x;
    const unsigned thread_layout_i = thread_idx / kThreadLayoutN;
    const unsigned thread_layout_j = thread_idx % kThreadLayoutN;
    // 在共享内存中初始化屏障 (只需一个线程初始化)
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier;
    if (thread_idx == 0) {
        init(&barrier, num_thread); // 参与同步的线程数
    }
    __syncthreads(); //同步
    T r_a[kThreadM];
    T r_b[kThreadN];
    T r_c[kThreadM][kThreadN] = {0};
    // first g->s
    load<T, kBlockM, kBlockN, kBlockK, VecType, kNumPerVec>(
        a, b, N, K, s_a[0], s_b[0],
        block_x, block_y, num_thread, thread_idx,
        barrier, 0);
    barrier.arrive_and_wait();
    // if (block_x == 0 && block_y == 0 && thread_idx == 0) {
    //     printf("first load.%f,%f\n", s_a[0][0][0], s_b[0][0][0]);
    // }
    bool flag = true;
    for (unsigned block_k_start = kBlockK; block_k_start < K; block_k_start += kBlockK) {
        // g->s
        load<T, kBlockM, kBlockN, kBlockK,
            VecType, kNumPerVec>(
            a, b, N, K, s_a[flag], s_b[flag],
            block_x, block_y, num_thread, thread_idx,
            barrier, block_k_start);
        // if (block_x == 0 && block_y == 0 && thread_idx == 0) {
        //     printf("block k:%u\n", block_k_start);
        // }
        // compute
        flag = !flag;
        compute<T, kBlockM, kBlockN, kBlockK, kThreadM, kThreadN,
            kThreadLayoutM, kThreadLayoutN>(s_a[flag], s_b[flag], r_a, r_b, r_c,
                                            thread_layout_i, thread_layout_j);
        barrier.arrive_and_wait();
        __syncthreads();
        // if (block_x == 0 && block_y == 0 && thread_idx == 0) {
        //     printf("block k: %u;%f %f %f\n", block_k_start, s_a[flag][0][0], s_b[flag][0][0], r_c[0][0]);
        // }
    }
    // if (block_x == 0 && block_y == 0 && thread_idx == 0) {
    //     printf("circle finish\n");
    // }
    flag = !flag;
    compute<T, kBlockM, kBlockN, kBlockK, kThreadM, kThreadN,
        kThreadLayoutM, kThreadLayoutN>(s_a[flag], s_b[flag], r_a, r_b, r_c,
                                        thread_layout_i, thread_layout_j);
    // if (block_x == 0 && block_y == 0 && thread_idx == 0) {
    //     printf("last compute;%f\n", r_c[0][0]);
    // }
    // r->g
#pragma unroll
    for (unsigned i = 0; i < kThreadM; ++i) {
#pragma unroll
        for (unsigned j = 0; j < kThreadN; ++j) {
            const unsigned row_idx = kBlockM * block_y + thread_layout_i + i * kThreadLayoutM;
            const unsigned col_idx = kBlockN * block_x + thread_layout_j + j * kThreadLayoutN;
            c[row_idx * N + col_idx] = r_c[i][j];
        }
    }
}