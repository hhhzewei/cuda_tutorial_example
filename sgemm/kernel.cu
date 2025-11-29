//
// Created by hzw on 2025/10/25.
//
// Kernel function to add the elements of two arrays
#include "kernel.h"
#include <mma.h>
__global__ void sgemm_naive(const unsigned M, const unsigned K, const unsigned N, const float *a, const float *b,
                            float *ret) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < N && y < M) {
        float value = 0.0f;
        for (unsigned i = 0; i < K; ++i) {
            value += a[y * K + i] * b[i * N + x];
        }
        ret[y * N + x] = value;
    }
}

__global__ void sgemm_tensor_core_v0(const unsigned K, const unsigned N, const half *a, const half *b, float *ret) {
    using namespace nvcuda;
    const unsigned warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize,
            warp_N = N / WMMA_TILE_N,
            warp_i = warpIdx / warp_N, warp_j = warpIdx % warp_N;

    wmma::fragment<wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::row_major> frag_b;
    wmma::fragment<wmma::accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, float> frag_c;

    wmma::fill_fragment(frag_c, 0.0f);
#pragma unroll
    for (int k = 0; k < K; k += WMMA_TILE_K) {
        wmma::load_matrix_sync(frag_a, &_2D_2_1D(a, warp_i * WMMA_TILE_M, k, K), K);
        wmma::load_matrix_sync(frag_b, &_2D_2_1D(b, k, warp_j * WMMA_TILE_N, N), N);
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    wmma::store_matrix_sync(&_2D_2_1D(ret, warp_i * WMMA_TILE_M, warp_j * WMMA_TILE_N, N), frag_c, N,
                            wmma::mem_row_major);
}
