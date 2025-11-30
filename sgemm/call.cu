//
// Created by hzw on 2025/11/4.
//
#include "call.h"
#include <util/util.h>
#include <cublas_v2.h>

void call_sgemm_naive(const unsigned M, const unsigned K, const unsigned N, const float *dev_a, const float *dev_b,
                      float *dev_ret) {
    dim3 blockDim(WARP_SIZE, WARP_SIZE), gridDim(CEIL(N, WARP_SIZE),CEIL(M, WARP_SIZE));
    sgemm_naive<<<gridDim,blockDim>>>(M, K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
}

void call_sgemm_block_tile(const unsigned M, const unsigned K, const unsigned N, const float *dev_a, const float *dev_b,
                           float *dev_ret) {
    // kernel
    dim3 blockDim(WARP_SIZE, WARP_SIZE), gridDim(CEIL(N, WARP_SIZE),CEIL(M, WARP_SIZE));
    sgemm_block_tile<WARP_SIZE, WARP_SIZE, WARP_SIZE><<<gridDim, blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
}

void call_sgemm_thread_tile_v0(const unsigned M, const unsigned K, const unsigned N, const float *dev_a, const float *dev_b,
                               float *dev_ret) {
    // kernel
    constexpr unsigned thread_m = 2, thread_n = 2;
    dim3 blockDim(CEIL(WARP_SIZE, thread_n), CEIL(WARP_SIZE, thread_m)),
            gridDim(CEIL(N, WARP_SIZE), CEIL(M, WARP_SIZE));
    sgemm_thread_tile_v0<WARP_SIZE, WARP_SIZE, WARP_SIZE, thread_m, thread_n><<<gridDim,blockDim>>>(
        K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
}


void call_sgemm_cublas(const unsigned M, const unsigned K, const unsigned N, const float *dev_a, const float *dev_b,
                       float *dev_ret) {
    // kernel
    cublasHandle_t handle{};
    cublasCreate(&handle);
    constexpr float alpha = 1.0, beta = 0.0;
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                dev_b, N,
                dev_a, K,
                &beta,
                dev_ret, N
    );
}

void call_sgemm_tensor_core_v0(const unsigned M, const unsigned K, const unsigned N, const float *dev_a, const float *dev_b,
                               float *dev_ret) {
    // kernel
    sgemm_tensor_core_v0<<<M * N / WMMA_TILE_M / WMMA_TILE_N, WARP_SIZE>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
}

void call_sgemm_tensor_core_v1(const unsigned M, const unsigned K, const unsigned N, const float *dev_a, const float *dev_b,
                               float *dev_ret) {
    // kernel
    // warp在block内按每行若干warp
    constexpr unsigned TILE_M = 64, TILE_N = 64;
    dim3 blockDim{TILE_N / WMMA_TILE_N * WARP_SIZE, TILE_M / WMMA_TILE_M};
    dim3 gridDim{N / TILE_N, M / TILE_M};
    sgemm_tensor_core_v1<TILE_M, TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
}
