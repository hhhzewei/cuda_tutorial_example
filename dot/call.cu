//
// Created by hzw on 2025/11/4.
//
#include "call.h"
#include <cublas_v2.h>

void call_dot_cublas(const unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret) {
    // init
    cudaMemset(dev_ret, 0, sizeof(float));
    // kernel
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSdot(handle, N, dev_a, 1, dev_b, 1, dev_ret);
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
}
