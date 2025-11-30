//
// Created by hzw on 2025/11/4.
//
#include "call.h"
#include "util/util.h"
#include <cublas_v2.h>

void prepare_elementwise(const unsigned N, const float *a, const float *b, float *&dev_a, float *&dev_b, float *&dev_ret) {\
    // device memory malloc
    cudaMalloc(&dev_a, N * sizeof(float));
    cudaMalloc(&dev_b, N * sizeof(float));
    cudaMalloc(&dev_ret, N * sizeof(float));
    // copy input
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
}

void call_add_cublas(const unsigned N, const float *a, const float *b, float *ret) {
    // device memory malloc
    float *dev_a, *dev_b, *dev_ret;
    prepare_elementwise(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    cublasHandle_t handle{};
    cublasCreate(&handle);
    constexpr float alpha = 1.0f; // 纯加法
    cublasSaxpy(handle, N, &alpha, dev_a, 1, dev_b, 1);
    // copy output
    cudaMemcpy(ret, dev_ret, N * sizeof(float), cudaMemcpyDeviceToHost);
    // device free
    batch_cuda_free({dev_a, dev_b, dev_ret});
}