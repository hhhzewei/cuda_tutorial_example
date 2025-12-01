//
// Created by hzw on 2025/11/4.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CALL_H
#define CUDA_TUTORIAL_EXAMPLE_CALL_H

#include <iostream>

#include "util/util.h"
#include "kernel.h"

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot(const unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret, float *a) {
    // init
    cudaMemset(dev_ret, 0, sizeof(float));
    // kernel
    dot<<<BLOCK_NUM, THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    // 恢复被更改的输入
    cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot_shared(const unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret) {
    // init
    cudaMemset(dev_ret, 0, sizeof(float));
    // kernel
    dot_share<BLOCK_NUM><<<BLOCK_NUM, THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot_shared_external(const unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret) {
    // init
    cudaMemset(dev_ret, 0, sizeof(float));
    // kernel
    dot_shared_external<<<BLOCK_NUM, THREAD_NUM,BLOCK_NUM * sizeof(float)>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot_warp_shuffle_v0(const unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret) {
    // init
    cudaMemset(dev_ret, 0, sizeof(float));
    // kernel
    constexpr unsigned warp_num = CEIL(THREAD_NUM, WARP_SIZE);
    dot_warp_shuffle_v0<warp_num><<<BLOCK_NUM, THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot_warp_shuffle_v1(const unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret) {
    // init
    cudaMemset(dev_ret, 0, sizeof(float));
    // kernel
    constexpr unsigned warp_num = CEIL(THREAD_NUM, WARP_SIZE);
    dot_warp_shuffle_v1<warp_num><<<BLOCK_NUM, THREAD_NUM>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
}

void call_dot_cublas(unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret);
#endif //CUDA_TUTORIAL_EXAMPLE_CALL_H
