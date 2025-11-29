//
// Created by hzw on 2025/11/4.
//
#include "call.h"
#include <cublas_v2.h>

void prepare_reduce(const unsigned N, const float *a, const float *b, float *&dev_a, float *&dev_b, float *&dev_ret) {
    // device memory malloc
    cudaMalloc(&dev_a, N * sizeof(float));
    cudaMalloc(&dev_b, N * sizeof(float));
    cudaMalloc(&dev_ret, sizeof(float));
    // copy input
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
}

void call_dot_cublas(const unsigned N, const float *a, const float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_reduce(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSdot(handle, N, dev_a, 1, dev_b, 1, dev_ret);
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}
