//
// Created by hzw on 2025/11/4.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CALL_H
#define CUDA_TUTORIAL_EXAMPLE_CALL_H

#include "util/util.h"
#include "kernel.h"

void prepare_elementwise(unsigned N, const float *a, const float *b, float *&dev_a, float *&dev_b, float *&dev_ret);

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_add(const unsigned N, float *a, float *b, float *ret) {
    // device memory malloc
    float *dev_a, *dev_b, *dev_ret;
    prepare_elementwise(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    add<<<BLOCK_NUM,THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, N * sizeof(float), cudaMemcpyDeviceToHost);
    // device free
    batch_cuda_free({dev_a, dev_b, dev_ret});
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_add_float4(const unsigned N, float *a, float *b, float *ret) {
    // device memory malloc
    float *dev_a, *dev_b, *dev_ret;
    prepare_elementwise(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    add_float4<<<BLOCK_NUM,THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, N * sizeof(float), cudaMemcpyDeviceToHost);
    // device free
    batch_cuda_free({dev_a, dev_b, dev_ret});
}

void call_add_cublas(unsigned N, const float *a, const float *b, float *ret);
#endif //CUDA_TUTORIAL_EXAMPLE_CALL_H
