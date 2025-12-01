//
// Created by hzw on 2025/11/4.
//
#include "call.h"
#include "kernel.h"
#include "util/util.h"
#include <cublas_v2.h>


void call_transpose_naive(const unsigned M, const unsigned N, float *dev_input, float *dev_output,
                          float *output) {
    // kernel
    constexpr unsigned warp_size = 32;
    dim3 blockDim(warp_size, warp_size), gridDim(CEIL(N, warp_size),CEIL(M, warp_size));
    transpose_naive<<<gridDim,blockDim>>>(M, N, dev_input, dev_output);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(output, dev_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}


void call_transpose_shared(const unsigned M, const unsigned N, float *dev_input, float *dev_output,
                           float *output) {
    // kernel
    dim3 blockDim(WARP_SIZE, WARP_SIZE), gridDim(CEIL(N, WARP_SIZE),CEIL(M, WARP_SIZE));
    transpose_shared<<<gridDim,blockDim>>>(M, N, dev_input, dev_output);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(output, dev_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_transpose_padding(const unsigned M, const unsigned N, float *dev_input, float *dev_output,
                            float *output) {
    // kernel
    dim3 blockDim(WARP_SIZE, WARP_SIZE), gridDim(CEIL(N, WARP_SIZE),CEIL(M, WARP_SIZE));
    transpose_padding<<<gridDim,blockDim>>>(M, N, dev_input, dev_output);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(output, dev_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_transpose_swizzle(const unsigned M, const unsigned N, float *dev_input, float *dev_output,
                            float *output) {
    // kernel
    dim3 blockDim(WARP_SIZE, WARP_SIZE), gridDim(CEIL(N, WARP_SIZE),CEIL(M, WARP_SIZE));
    transpose_swizzle<<<gridDim,blockDim>>>(M, N, dev_input, dev_output);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(output, dev_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_transpose_cubalas(const unsigned M, const unsigned N, float *dev_input, float *dev_output,
                            float *output) {
    // kernel
    cublasHandle_t handle{};
    cublasCreate(&handle);
    constexpr float alpha = 1.0f, beta = 0.0f;
    float *dev_tmp;
    cudaMalloc(&dev_tmp, M * N * sizeof(float));
    cublasSgeam(handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                M, N,
                &alpha,
                dev_input, N,
                &beta,
                dev_tmp, M,
                dev_output, M);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(output, dev_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // 销毁临时内存
    cudaFree(dev_tmp);
}
