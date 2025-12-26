//
// Created by hzw on 2025/11/4.
//
#include "call.h"
#include <util/util.h>
#include <cublas_v2.h>
// #include "cutlass/gemm/device/gemm.h"
// #include "cutlass/util/device_memory.h"

void call_sgemm_cublas(const unsigned M, const unsigned K, const unsigned N, const float *dev_a, const float *dev_b,
                       float *dev_ret, float *ret) {
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
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// void call_sgemm_cutlass(unsigned M, unsigned K, unsigned N, const float *dev_a, const float *dev_b, float *dev_c,
//                         float *c) {
//     // 1. 定义 GEMM 类型（同前例，假设使用 Simt 架构以获得通用性）
//     using Gemm = cutlass::gemm::device::Gemm<
//         float, cutlass::layout::RowMajor,
//         float, cutlass::layout::RowMajor,
//         float, cutlass::layout::RowMajor
//     >;
//
//     // 2. 准备 CUTLASS 参数
//     // 注意：我们将 dev_c 同时作为输入(C)和输出(D)
//     Gemm::Arguments args(
//         {(int) M, (int) N, (int) K},
//         {dev_a, K}, // A (M x K)
//         {dev_b, N}, // B (K x N)
//         {dev_c, N}, // C (M x N) - 源矩阵
//         {dev_c, N}, // D (M x N) - 结果矩阵，覆盖 dev_c
//         {1.0f, 0.0f}
//     );
//
//     // 3. 运行内核
//     Gemm gemm_op;
//     cutlass::Status status = gemm_op(args);
//
//     if (status != cutlass::Status::kSuccess) {
//         throw std::runtime_error("CUTLASS kernel failed.");
//     }
//
//     // 4. 数据拷回 (显式同步)
//     // 因为 CUTLASS 内核在默认流上是异步的，cudaMemcpy 会触发同步并拷贝数据
//     size_t size_c = sizeof(float) * M * N;
//     cudaError_t err = cudaMemcpy(c, dev_c, size_c, cudaMemcpyDeviceToHost);
//
//     if (err != cudaSuccess) {
//         throw std::runtime_error("CUDA Copy back failed.");
//     }
// }

void call_sgemm_naive(const unsigned M, const unsigned K, const unsigned N, const float *dev_a, const float *dev_b,
                      float *dev_ret, float *ret) {
    constexpr unsigned THREAD_NUM = 256, BLOCK_PER_SM = NUM_THREA_PER_SM / THREAD_NUM;
    sgemm_naive<<<NUM_SM * BLOCK_PER_SM,THREAD_NUM>>>(M, K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_sgemm_block_tile(const unsigned M, const unsigned K, const unsigned N, const float *dev_a, const float *dev_b,
                           float *dev_ret, float *ret) {
    // kernel
    dim3 blockDim(WARP_SIZE, WARP_SIZE), gridDim(CEIL(N, WARP_SIZE),CEIL(M, WARP_SIZE));
    sgemm_block_tile<WARP_SIZE, WARP_SIZE, WARP_SIZE><<<gridDim, blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_sgemm_thread_tile_v0(const unsigned M, const unsigned K, const unsigned N, const float *dev_a,
                               const float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    constexpr unsigned thread_m = 2, thread_n = 2;
    dim3 blockDim(CEIL(WARP_SIZE, thread_n), CEIL(WARP_SIZE, thread_m)),
            gridDim(CEIL(N, WARP_SIZE), CEIL(M, WARP_SIZE));
    sgemm_thread_tile_v0<WARP_SIZE, WARP_SIZE, WARP_SIZE, thread_m, thread_n><<<gridDim,blockDim>>>(
        K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}


void call_sgemm_tensor_core_v0(const unsigned M, const unsigned K, const unsigned N, const float *dev_a,
                               const float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    constexpr unsigned WMMA_TILE_M = 16, WMMA_TILE_N = 16;
    sgemm_tensor_core_v0<<<M * N / WMMA_TILE_M / WMMA_TILE_N, WARP_SIZE>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_sgemm_tensor_core_v1(const unsigned M, const unsigned K, const unsigned N, const float *dev_a,
                               const float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    // warp在block内按每行若干warp
    constexpr unsigned WMMA_TILE_M = 16, WMMA_TILE_N = 16;
    constexpr unsigned TILE_M = 64, TILE_N = 64;
    constexpr unsigned NUM_WARP = CEIL(TILE_N, WMMA_TILE_N) * CEIL(TILE_M, WMMA_TILE_M);
    constexpr unsigned blockDim = NUM_WARP * WARP_SIZE;
    dim3 gridDim{N / TILE_N, M / TILE_M};
    sgemm_tensor_core_v1<TILE_M, TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_sgemm_tensor_core_v2(const unsigned M, const unsigned K, const unsigned N, const float *dev_a,
                               const float *dev_b,
                               float *dev_ret, float *ret) {
    // kernel
    // warp在block内按每行若干warp
    constexpr unsigned WMMA_TILE_M = 16, WMMA_TILE_N = 16;
    constexpr unsigned TILE_M = 64, TILE_N = 64;
    constexpr unsigned NUM_WARP = CEIL(TILE_N, WMMA_TILE_N) * CEIL(TILE_M, WMMA_TILE_M);
    constexpr unsigned blockDim = NUM_WARP * WARP_SIZE;
    dim3 gridDim{N / TILE_N, M / TILE_M};
    sgemm_tensor_core_v2<TILE_M, TILE_N><<<gridDim,blockDim>>>(K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}
