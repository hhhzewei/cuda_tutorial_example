//
// Created by hzw on 2025/11/4.
//
#include "call.h"
#include <util/util.h>
#include <cublas_v2.h>

void prepare_sgemm(const unsigned M, const unsigned K, const unsigned N,
                   const float *a, const float *b, float *&dev_a, float *&dev_b, float *&dev_ret) {
    cudaStream_t stream_a, stream_b, stream_ret;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);
    cudaStreamCreate(&stream_ret);
    // 创建固定内存
    cudaMallocAsync(&dev_a, M * K * sizeof(float), stream_a);
    cudaMallocAsync(&dev_b, N * K * sizeof(float), stream_b);
    cudaMallocAsync(&dev_ret, M * N * sizeof(float), stream_ret);
    // copy input
    cudaMemcpyAsync(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice, stream_a);
    cudaMemcpyAsync(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice, stream_b);
}

void prepare_sgemm_tensor_core(const unsigned M, const unsigned K, const unsigned N,
                               const float *a, const float *b, half *&dev_a, half *&dev_b, float *&dev_ret) {
    half *half_a = (half *) malloc(sizeof(half) * M * K),
            *half_b = (half *) malloc(sizeof(half) * K * N);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            _2D_2_1D(half_a, i, j, K) = half(_2D_2_1D(a, i, j, K));
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            _2D_2_1D(half_b, i, j, N) = half(_2D_2_1D(b, i, j, N));
        }
    }
    cudaStream_t stream_a, stream_b, stream_ret;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);
    cudaStreamCreate(&stream_ret);
    // 创建device内存
    cudaMallocAsync(&dev_a, M * K * sizeof(half), stream_a);
    cudaMallocAsync(&dev_b, N * K * sizeof(half), stream_b);
    cudaMallocAsync(&dev_ret, M * N * sizeof(float), stream_ret);
    // copy input
    cudaMemcpyAsync(dev_a, half_a, M * K * sizeof(half), cudaMemcpyHostToDevice, stream_a);
    cudaMemcpyAsync(dev_b, half_b, K * N * sizeof(half), cudaMemcpyHostToDevice, stream_b);
    // free half
    free(half_a);
    free(half_b);
}


void call_sgemm_naive(const unsigned M, const unsigned K, const unsigned N, const float *a, const float *b,
                      float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_sgemm(M, K, N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    constexpr unsigned warp_size = 32;
    dim3 blockDim(warp_size, warp_size), gridDim(CEIL(N, warp_size),CEIL(M, warp_size));
    sgemm_naive<<<gridDim,blockDim>>>(M, K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

void call_sgemm_block_tile(const unsigned M, const unsigned K, const unsigned N, const float *a, const float *b,
                           float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_sgemm(M, K, N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    constexpr unsigned warp_size = 32;
    dim3 blockDim(warp_size, warp_size), gridDim(CEIL(N, warp_size),CEIL(M, warp_size));
    sgemm_block_tile<warp_size, warp_size, warp_size><<<gridDim, blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

void call_sgemm_thread_tile_v0(const unsigned M, const unsigned K, const unsigned N, const float *a, const float *b,
                               float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_sgemm(M, K, N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    constexpr unsigned warp_size = 32,
            thread_m = 2, thread_n = 2;
    dim3 blockDim(CEIL(warp_size, thread_n), CEIL(warp_size, thread_m)),
            gridDim(CEIL(N, warp_size), CEIL(M, warp_size));
    sgemm_thread_tile_v0<warp_size, warp_size, warp_size, thread_m, thread_n><<<gridDim,blockDim>>>(
        K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}


void call_sgemm_cublas(const unsigned M, const unsigned K, const unsigned N, const float *a, const float *b,
                       float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_sgemm(M, K, N, a, b, dev_a, dev_b, dev_ret);
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
    // copy output
    // cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

void call_sgemm_tensor_core_v0(const unsigned M, const unsigned K, const unsigned N, const float *a, const float *b,
                               float *ret) {
    // device memory
    half *dev_a, *dev_b;
    float *dev_ret;
    prepare_sgemm_tensor_core(M, K, N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    sgemm_tensor_core_v0<<<M * N / WMMA_TILE_M / WMMA_TILE_N, WARP_SIZE>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

void call_sgemm_tensor_core_v1(const unsigned M, const unsigned K, const unsigned N, const float *a, const float *b,
                               float *ret) {
    // device memory
    half *dev_a, *dev_b;
    float *dev_ret;
    prepare_sgemm_tensor_core(M, K, N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    // warp在block内按每行若干warp
    constexpr unsigned TILE_M = 64, TILE_N = 64;
    dim3 blockDim{TILE_N / WMMA_TILE_N * WARP_SIZE, TILE_M / WMMA_TILE_M};
    dim3 gridDim{N / TILE_N, M / TILE_M};
    sgemm_tensor_core_v1<TILE_M, TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}
