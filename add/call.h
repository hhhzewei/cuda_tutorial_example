//
// Created by hzw on 2025/11/4.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CALL_H
#define CUDA_TUTORIAL_EXAMPLE_CALL_H

#include "util/util.h"
#include "kernel.h"

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_add(const unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret) {
    // kernel
    add<<<BLOCK_NUM,THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // memcpy
    cudaMemcpy(ret, dev_ret, N * sizeof(float), cudaMemcpyDeviceToHost);
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_add_float4(const unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret) {
    // kernel
    add_float4<<<BLOCK_NUM,THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // memcpy
    cudaMemcpy(ret, dev_ret, N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_add_cublas(unsigned N, float *dev_a, float *dev_b, float *ret, float *b);
#endif //CUDA_TUTORIAL_EXAMPLE_CALL_H
