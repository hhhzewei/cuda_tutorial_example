//
// Created by hzw on 2026/5/4.
//
#pragma once

template<typename T>
struct VecType128b;

template<>
struct VecType128b<float> {
    using Vec = float4;
};

template<
    typename T,
    unsigned kBlockM, unsigned kBlockN, unsigned kBlockK, unsigned kThreadM, unsigned kThreadN>
__global__ void gemm(const T *a, const T *b, T *c,
                     unsigned M, const unsigned N, const unsigned K) {
    constexpr unsigned kNumPer128b = 16 / sizeof(T);
    using Vec128b = typename VecType128b<T>::Vec;
    __shared__ T s_a[2][kBlockM][kBlockK];
    __shared__ T s_b[2][kBlockK][kBlockN];
    const unsigned thread_idx = threadIdx.x;
    const unsigned block_x = blockIdx.x;
    const unsigned block_y = blockIdx.y;
    const unsigned num_thread = blockDim.x * blockDim.y;
    const unsigned thread_layout_m = num_thread / 32;
    constexpr unsigned thread_layout_n = 32;
    const unsigned thread_layout_i = thread_idx / thread_layout_n;
    const unsigned thread_layout_j = thread_idx % thread_layout_n;
    bool flag = false;
    T r_a[kThreadM];
    T r_b[kThreadN];
    T r_c[kThreadM][kThreadN] = {0};
    // read
    for (unsigned block_k_start = 0; block_k_start < K; block_k_start += kBlockK) {
        for (unsigned i = thread_idx * kNumPer128b; i < kBlockM * kBlockK; i += num_thread * kNumPer128b) {
            const unsigned s_row_idx = i / kBlockK;
            const unsigned s_col_idx = i % kBlockK;
            const unsigned g_row_idx = block_y * kBlockM + s_row_idx;
            const unsigned g_col_idx = block_k_start + s_col_idx;
            *reinterpret_cast<Vec128b *>(&s_a[flag][s_row_idx][s_col_idx]) =
                    *reinterpret_cast<const Vec128b *>(&a[g_row_idx * K + g_col_idx]);
        }
        for (unsigned i = thread_idx * kNumPer128b; i < kBlockN * kBlockK; i += num_thread * kNumPer128b) {
            const unsigned s_row_idx = i / kBlockN;
            const unsigned s_col_idx = i % kBlockN;
            const unsigned g_col_idx = block_x * kBlockN + s_col_idx;
            const unsigned g_row_idx = block_k_start + s_row_idx;
            *reinterpret_cast<Vec128b *>(&s_b[flag][s_row_idx][s_col_idx]) =
                    *reinterpret_cast<const Vec128b *>(&b[g_row_idx * N + g_col_idx]);
        }
        __syncthreads();
        // compute
#pragma unroll
        for (unsigned k = 0; k < kBlockK; ++k) {
#pragma unroll
            for (unsigned i = 0; i < kThreadM; ++i) {
                r_a[i] = s_a[flag][thread_layout_i + i * thread_layout_m][k];
            }
#pragma unroll
            for (unsigned i = 0; i < kThreadN; ++i) {
                r_b[i] = s_b[flag][k][thread_layout_j + i * thread_layout_n];
            }
#pragma unroll
            for (unsigned i = 0; i < kThreadM; ++i) {
#pragma unroll
                for (unsigned j = 0; j < kThreadN; ++j) {
                    r_c[i][j] += r_a[i] * r_b[j];
                }
            }
        }
        flag = !flag;
    }
    __shared__ T s_c[kBlockM][kBlockN];
#pragma unroll
    for (unsigned i = 0; i < kThreadM; ++i) {
#pragma unroll
        for (unsigned j = 0; j < kThreadN; ++j) {
            const unsigned row_idx = thread_layout_i + i * thread_layout_m;
            const unsigned col_idx = thread_layout_j + j * thread_layout_n;
            s_c[row_idx][col_idx] = r_c[i][j];
        }
    }
    __syncthreads();
    for (int i = thread_idx * kNumPer128b; i < kBlockM * kBlockN; i += num_thread * kNumPer128b) {
        const unsigned s_row_idx = i / kBlockN;
        const unsigned s_col_idx = i % kBlockN;
        const unsigned g_row_idx = block_y * kBlockM + s_row_idx;
        const unsigned g_col_idx = block_x * kBlockN + s_col_idx;
        *reinterpret_cast<Vec128b *>(c + g_row_idx * N + g_col_idx) =
                *reinterpret_cast<Vec128b *>(&s_c[s_row_idx][s_col_idx]);
    }
}
