//
// Created by hzw on 2025/11/4.
//
#include "call.h"


void call_transpose_naive(unsigned M, unsigned N, float *input, float *output) {
    // cuda malloc
    const unsigned SIZE = M * N * sizeof(float);
    float *dev_input, *dev_output;
    cudaMalloc(&dev_input, SIZE);
    cudaMalloc(&dev_output, SIZE);
    cudaMemcpy(dev_input, input, SIZE, cudaMemcpyHostToDevice);
    // kernel
    constexpr unsigned warp_size = 32;
    dim3 blockDim(warp_size, warp_size), gridDim(CEIL(N, warp_size),CEIL(M, warp_size));
    transpose_naive<<<gridDim,blockDim>>>(M, N, dev_input, dev_output);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(output, dev_output, SIZE, cudaMemcpyDeviceToHost);
    // cuda free
    cudaFree(dev_input);
    cudaFree(dev_output);
}


void call_transpose_padding(unsigned M, unsigned N, float *input, float *output) {
    // cuda malloc
    const unsigned SIZE = M * N * sizeof(float);
    float *dev_input, *dev_output;
    cudaMalloc(&dev_input, SIZE);
    cudaMalloc(&dev_output, SIZE);
    cudaMemcpy(dev_input, input, SIZE, cudaMemcpyHostToDevice);
    // kernel
    constexpr unsigned warp_size = 32;
    dim3 blockDim(warp_size, warp_size), gridDim(CEIL(N, warp_size),CEIL(M, warp_size));
    transpose_padding<warp_size><<<gridDim,blockDim>>>(M, N, dev_input, dev_output);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(output, dev_output, SIZE, cudaMemcpyDeviceToHost);
    // cuda free
    cudaFree(dev_input);
    cudaFree(dev_output);
}

void call_transpose_swizzle(unsigned M, unsigned N, float *input, float *output) {
    // cuda malloc
    const unsigned SIZE = M * N * sizeof(float);
    float *dev_input, *dev_output;
    cudaMalloc(&dev_input, SIZE);
    cudaMalloc(&dev_output, SIZE);
    cudaMemcpy(dev_input, input, SIZE, cudaMemcpyHostToDevice);
    // kernel
    constexpr unsigned warp_size = 32;
    dim3 blockDim(warp_size, warp_size), gridDim(CEIL(N, warp_size),CEIL(M, warp_size));
    transpose_swizzle<warp_size><<<gridDim,blockDim>>>(M, N, dev_input, dev_output);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(output, dev_output, SIZE, cudaMemcpyDeviceToHost);
    // cuda free
    cudaFree(dev_input);
    cudaFree(dev_output);
}

void call_sgemm_naive(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    cudaMalloc(&dev_a, M * K * sizeof(float));
    cudaMalloc(&dev_b, N * K * sizeof(float));
    cudaMalloc(&dev_ret, M * N * sizeof(float));
    // copy input
    cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    // kernel
    constexpr unsigned warp_size = 32;
    dim3 blockDim(warp_size, warp_size), gridDim(CEIL(N, warp_size),CEIL(M, warp_size));
    sgemm_naive<<<gridDim,blockDim>>>(M, K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_ret);
}

void call_sgemm_block_tile(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    cudaMalloc(&dev_a, M * K * sizeof(float));
    cudaMalloc(&dev_b, N * K * sizeof(float));
    cudaMalloc(&dev_ret, M * N * sizeof(float));
    // copy input
    cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    // kernel
    constexpr unsigned warp_size = 32;
    dim3 blockDim(warp_size, warp_size), gridDim(CEIL(N, warp_size),CEIL(M, warp_size));
    sgemm_block_tile<warp_size, warp_size, warp_size><<<gridDim, blockDim>>>(M, K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_ret);
}

void call_sgemm_thread_tile(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    cudaMalloc(&dev_a, M * K * sizeof(float));
    cudaMalloc(&dev_b, K * N * sizeof(float));
    cudaMalloc(&dev_ret, M * N * sizeof(float));
    // copy input
    cudaMemcpy(dev_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);
    // kernel
    constexpr unsigned warp_size = 32,
            thread_m = 2, thread_n = 2;
    dim3 blockDim(CEIL(warp_size, thread_n), CEIL(warp_size, thread_m)),
            gridDim(CEIL(N, warp_size), CEIL(M, warp_size));
    sgemm_thread_tile<warp_size, warp_size, warp_size, thread_m, thread_n><<<gridDim,blockDim>>>(
        M, K, N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_ret);
}
