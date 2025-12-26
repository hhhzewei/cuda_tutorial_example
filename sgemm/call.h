//
// Created by hzw on 2025/11/4.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CALL_H
#define CUDA_TUTORIAL_EXAMPLE_CALL_H

#include "util/util.h"
#include "kernel.h"

void call_sgemm_cublas(unsigned M, unsigned K, unsigned N, const float *dev_a, const float *dev_b, float *dev_ret,
                       float *ret);

// void call_sgemm_cutlass(unsigned M, unsigned K, unsigned N, const float *dev_a, const float *dev_b, float *dev_c,
//                         float *c);

void call_sgemm_naive(unsigned M, unsigned K, unsigned N, const float *dev_a, const float *dev_b, float *dev_ret,
                      float *ret);

void call_sgemm_block_tile(unsigned M, unsigned K, unsigned N, const float *dev_a, const float *dev_b, float *dev_ret,
                           float *ret);

void call_sgemm_thread_tile_v0(unsigned M, unsigned K, unsigned N, const float *dev_a, const float *dev_b,
                               float *dev_ret, float *ret);

template<unsigned BLOCK_M, unsigned BLOCK_N,
    unsigned TILE_K,
    unsigned THREAD_M, unsigned THREAD_N,
    unsigned WARP_CIRCE_LOG2>
void call_sgemm_thread_tile_v1(const unsigned M, const unsigned K, const unsigned N, float *dev_a, float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    dim3 blockDim(BLOCK_N, BLOCK_M),
            gridDim(CEIL(N, BLOCK_N * THREAD_N), CEIL(M, BLOCK_M * THREAD_M));
    sgemm_thread_tile_v1<BLOCK_M, BLOCK_N, TILE_K,
        THREAD_M, THREAD_N,
        WARP_CIRCE_LOG2><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

template<unsigned TILE_M = 32, unsigned TILE_K = 32, unsigned TILE_N = 32,
    unsigned THREAD_TILE_M = 2, unsigned THREAD_TILE_N = 2>
void call_sgemm_thread_tile_v2(const unsigned M, const unsigned K, const unsigned N, float *dev_a, float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    constexpr unsigned BLOCK_M = TILE_M / THREAD_TILE_M, BLOCK_N = TILE_N / THREAD_TILE_N;
    dim3 blockDim(BLOCK_N, BLOCK_M),
            gridDim(CEIL(N, TILE_N), CEIL(M, BLOCK_M * THREAD_TILE_M));
    sgemm_thread_tile_v2<TILE_M, TILE_N,
        TILE_K,
        THREAD_TILE_M, THREAD_TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

template<unsigned TILE_M = 128, unsigned TILE_K = 8, unsigned TILE_N = 128,
    unsigned THREAD_TILE_M = 8, unsigned THREAD_TILE_N = 8>
void call_sgemm_thread_tile_v3(const unsigned M, const unsigned K, const unsigned N, float *dev_a, float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    constexpr unsigned BLOCK_M = TILE_M / THREAD_TILE_M, BLOCK_N = TILE_N / THREAD_TILE_N;
    dim3 blockDim(BLOCK_N, BLOCK_M),
            gridDim(CEIL(N, TILE_N), CEIL(M, BLOCK_M * THREAD_TILE_M));
    sgemm_thread_tile_v3<TILE_M, TILE_K, TILE_N,
        THREAD_TILE_M, THREAD_TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

template<unsigned TILE_M = 128, unsigned TILE_K = 8, unsigned TILE_N = 128,
    unsigned THREAD_TILE_M = 8, unsigned THREAD_TILE_N = 8>
void call_sgemm_thread_tile_v4(const unsigned M, const unsigned K, const unsigned N, float *dev_a, float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    constexpr unsigned BLOCK_M = TILE_M / THREAD_TILE_M, BLOCK_N = TILE_N / THREAD_TILE_N;
    dim3 blockDim(BLOCK_N, BLOCK_M),
            gridDim(CEIL(N, TILE_N), CEIL(M, BLOCK_M * THREAD_TILE_M));
    sgemm_thread_tile_v4<TILE_M, TILE_K, TILE_N,
        THREAD_TILE_M, THREAD_TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}


template<unsigned TILE_M = 128, unsigned TILE_K = 8, unsigned TILE_N = 128,
    unsigned THREAD_TILE_M = 8, unsigned THREAD_TILE_N = 8>
void call_sgemm_thread_tile_v5(const unsigned M, const unsigned K, const unsigned N, float *dev_a, float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    constexpr unsigned BLOCK_M = TILE_M / THREAD_TILE_M, BLOCK_N = TILE_N / THREAD_TILE_N;
    dim3 blockDim(BLOCK_N, BLOCK_M),
            gridDim(CEIL(N, TILE_N), CEIL(M, BLOCK_M * THREAD_TILE_M));
    sgemm_thread_tile_v5<TILE_M, TILE_K, TILE_N,
        THREAD_TILE_M, THREAD_TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

template<unsigned TILE_M = 128, unsigned TILE_K = 8, unsigned TILE_N = 128,
    unsigned THREAD_TILE_M = 8, unsigned THREAD_TILE_N = 8>
void call_sgemm_thread_tile_v6(const unsigned M, const unsigned K, const unsigned N, float *dev_a, float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    constexpr unsigned BLOCK_M = TILE_M / THREAD_TILE_M, BLOCK_N = TILE_N / THREAD_TILE_N;
    dim3 blockDim(BLOCK_N, BLOCK_M),
            gridDim(CEIL(N, TILE_N), CEIL(M, BLOCK_M * THREAD_TILE_M));
    sgemm_thread_tile_v6<TILE_M, TILE_K, TILE_N,
        THREAD_TILE_M, THREAD_TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// template<unsigned TILE_M = 128, unsigned TILE_K = 8, unsigned TILE_N = 128,
//     unsigned THREAD_TILE_M = 8, unsigned THREAD_TILE_N = 8>
template<unsigned TILE_M = 64, unsigned TILE_K = 16, unsigned TILE_N = 64,
    unsigned THREAD_TILE_M = 4, unsigned THREAD_TILE_N = 4>// 设定tile尺寸减少读共享内存的bank conflict
void call_sgemm_thread_tile_v7(const unsigned M, const unsigned K, const unsigned N, float *dev_a, float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    constexpr unsigned BLOCK_M = TILE_M / THREAD_TILE_M, BLOCK_N = TILE_N / THREAD_TILE_N;
    dim3 blockDim(BLOCK_N, BLOCK_M),
            gridDim(CEIL(N, TILE_N), CEIL(M, TILE_M));
    sgemm_thread_tile_v7<TILE_M, TILE_K, TILE_N,
        THREAD_TILE_M, THREAD_TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_sgemm_tensor_core_v0(unsigned M, unsigned K, unsigned N, const float *dev_a, const float *dev_b,
                               float *dev_ret, float *ret);

void call_sgemm_tensor_core_v1(unsigned M, unsigned K, unsigned N, const float *dev_a, const float *dev_b,
                               float *dev_ret, float *ret);

void call_sgemm_tensor_core_v2(unsigned M, unsigned K, unsigned N, const float *dev_a, const float *dev_b,
                               float *dev_ret, float *ret);
#endif //CUDA_TUTORIAL_EXAMPLE_CALL_H
