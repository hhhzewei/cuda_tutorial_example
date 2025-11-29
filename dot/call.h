//
// Created by hzw on 2025/11/4.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CALL_H
#define CUDA_TUTORIAL_EXAMPLE_CALL_H

#include <iostream>

#include "util/util.h"
#include "kernel.h"

void prepare_reduce(unsigned N, const float *a, const float *b, float *&dev_a, float *&dev_b, float *&dev_ret);

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot(const unsigned N, float *a, float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_reduce(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    dot<<<BLOCK_NUM, THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot_shared(const unsigned N, float *a, float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_reduce(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    dot_share<BLOCK_NUM><<<BLOCK_NUM, THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot_shared_external(const unsigned N, float *a, float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_reduce(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    dot_shared_external<<<BLOCK_NUM, THREAD_NUM,BLOCK_NUM * sizeof(float)>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot_warp_shuffle_v0(const unsigned N, float *a, float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_reduce(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    constexpr unsigned warp_num = CEIL(THREAD_NUM, WARP_SIZE);
    dot_warp_shuffle_v0<warp_num><<<BLOCK_NUM, THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot_warp_shuffle_v1(const unsigned N, float *a, float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_reduce(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    constexpr unsigned warp_num = CEIL(THREAD_NUM, WARP_SIZE);
    dot_warp_shuffle_v1<warp_num><<<BLOCK_NUM, THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

void call_dot_cublas(unsigned N, const float *a, const float *b, float *ret);
#endif //CUDA_TUTORIAL_EXAMPLE_CALL_H
