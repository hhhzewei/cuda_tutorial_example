//
// Created by hzw on 2025/10/27.
//

#include "util.h"
#include <cstdio>

void check_error(const cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error:%s\n", cudaGetErrorString(err));
    }
}

void batch_free(const std::initializer_list<void *> ptr_list) {
    for (const auto p: ptr_list) {
        cudaFree(p);
    }
}

void destroy(std::initializer_list<destroy_param> param_list, const cudaEvent_t &kernel_finish) {
    // 记录kernel默认流
    cudaEventRecord(kernel_finish, nullptr);
    for (auto param: param_list) {
        // wait kernel finish
        cudaStreamWaitEvent(param.stream, kernel_finish);
        // cuda free
        cudaFreeAsync(param.dev_p, param.stream);
        // stream sync
        cudaStreamSynchronize(param.stream);
        // destroy stream
        cudaStreamDestroy(param.stream);
        // host free
        free(param.p);
    }
    // destroy event
    cudaEventDestroy(kernel_finish);
}