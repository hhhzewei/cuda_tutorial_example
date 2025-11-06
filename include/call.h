//
// Created by hzw on 2025/11/4.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CALL_H
#define CUDA_TUTORIAL_EXAMPLE_CALL_H

#include <iostream>

#include "check.h"
#include "kernel.h"

void prepare_elementwise(unsigned N, float *a, float *b, float *&dev_a, float *&dev_b, float *&dev_ret);

void prepare_reduce(unsigned N, float *a, float *b, float *&dev_a, float *&dev_b, float *&dev_ret);

void prepare_transpose(unsigned M, unsigned N, float *input, float *&dev_input, float *&dev_output);

void prepare_sgemm(unsigned M, unsigned K, unsigned N, float *a, float *b, float *&dev_a, float *&dev_b,
                   float *&dev_ret);

void batch_free(std::initializer_list<float *> ptr_list);

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_add(unsigned N, float *a, float *b, float *ret) {
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
    batch_free({dev_a, dev_b, dev_ret});
}

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_add_float4(unsigned N, float *a, float *b, float *ret) {
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
    batch_free({dev_a, dev_b, dev_ret});
}

void call_add_cublas(unsigned N, float *a, float *b, float *ret);

template<unsigned BLOCK_NUM, unsigned THREAD_NUM>
void call_dot(unsigned N, float *a, float *b, float *ret) {
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
void call_dot_shared(unsigned N, float *a, float *b, float *ret) {
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
void call_dot_shared_external(unsigned N, float *a, float *b, float *ret) {
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
void call_dot_warp_shuffle(unsigned N, float *a, float *b, float *ret) {
    // device memory
    float *dev_a, *dev_b, *dev_ret;
    prepare_reduce(N, a, b, dev_a, dev_b, dev_ret);
    // kernel
    constexpr unsigned warp_num = CEIL(THREAD_NUM, 32);
    dot_warp_shuffle<<<BLOCK_NUM, THREAD_NUM, warp_num * sizeof(float)>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
    // cuda free
    batch_free({dev_a, dev_b, dev_ret});
}

void call_dot_cublas(unsigned N, float *a, float *b, float *ret);

void call_transpose_naive(unsigned M, unsigned N, float *input, float *output);

void call_transpose_padding(unsigned M, unsigned N, float *input, float *output);

void call_transpose_swizzle(unsigned M, unsigned N, float *input, float *output);

void call_transpose_cubalas(unsigned M, unsigned N, float *input, float *output);

void call_sgemm_naive(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret);

void call_sgemm_block_tile(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret);

void call_sgemm_thread_tile(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret);

void call_sgemm_cublas(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret);
#endif //CUDA_TUTORIAL_EXAMPLE_CALL_H
