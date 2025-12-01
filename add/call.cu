//
// Created by hzw on 2025/11/4.
//
#include "call.h"
#include "util/util.h"
#include <cublas_v2.h>

void call_add_cublas(const unsigned N, float *dev_a, float *dev_b, float *ret, float *b) {
    // kernel
    cublasHandle_t handle{};
    cublasCreate(&handle);
    constexpr float alpha = 1.0f; // 纯加法
    cublasSaxpy(handle, N, &alpha, dev_a, 1, dev_b, 1);
    cudaMemcpy(ret, dev_b, N * sizeof(float), cudaMemcpyDeviceToHost);
    // 恢复被覆盖的数据
    cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
}
