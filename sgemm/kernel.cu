//
// Created by hzw on 2025/10/25.
//
// Kernel function to add the elements of two arrays
#include "kernel.h"
#include <mma.h>
__global__ void sgemm_naive(const unsigned M, const unsigned K, const unsigned N, const float *a, const float *b,
                            float *ret) {
    const unsigned threadIdxGlobal = blockIdx.x * blockDim.x + threadIdx.x,
            NUM_THREAD = gridDim.x * blockDim.x;
    for (unsigned i = threadIdxGlobal; i < M * N; i += NUM_THREAD) {
        const unsigned x = i / N, y = i % N;
        float value = 0.0f;
        for (unsigned k = 0; k < K; ++k) {
            value += a[x * K + k] * b[k * N + y];
        }
        ret[x * N + y] = value;
    }
}

__global__ void sgemm_tensor_core_v0(const unsigned K, const unsigned N, const float *a, const float *b, float *ret) {
    constexpr unsigned WMMA_TILE_M= 16, WMMA_TILE_K = 16, WMMA_TILE_N = 16;
    using namespace nvcuda;
    const unsigned warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize,
            warp_N = N / WMMA_TILE_N,
            warp_i = warpIdx / warp_N, warp_j = warpIdx % warp_N;
    wmma::fragment<wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);
    __shared__ half tile_a[WMMA_TILE_M][WMMA_TILE_K], tile_b[WMMA_TILE_K][WMMA_TILE_N];
#pragma unroll
    for (int k = 0; k < K; k += WMMA_TILE_K) {
        //load to shared
#pragma unroll
        for (unsigned i = threadIdx.x; i < WMMA_TILE_M * WMMA_TILE_K; i += blockDim.x) {
            const unsigned shared_i = i / WMMA_TILE_K, shared_j = i % WMMA_TILE_K;
            tile_a[shared_i][shared_j] = __float2half(_2D_2_1D(a, shared_i + warp_i * WMMA_TILE_M, k + shared_j, K));
        }
#pragma unroll
        for (unsigned i = threadIdx.x; i < WMMA_TILE_K * WMMA_TILE_N; i += blockDim.x) {
            const unsigned shared_i = i / WMMA_TILE_N, shared_j = i % WMMA_TILE_N;
            tile_b[shared_i][shared_j] = __float2half(_2D_2_1D(b, k + shared_i, shared_j + warp_j * WMMA_TILE_N, N));
        }
        wmma::load_matrix_sync(frag_a, &tile_a[0][0],WMMA_TILE_K);
        wmma::load_matrix_sync(frag_b, &tile_b[0][0],WMMA_TILE_N);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    wmma::store_matrix_sync(&_2D_2_1D(ret, warp_i * WMMA_TILE_M, warp_j * WMMA_TILE_N, N), frag_c, N,
                            wmma::mem_row_major);
}
