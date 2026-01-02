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

void call_dot_warp_shuffle_xor_v1(const unsigned N, float *dev_a, float *dev_b, float *dev_ret, float *ret) {
    // init
    cudaMemset(dev_ret, 0, sizeof(float));
    // kernel
    constexpr unsigned NUM_THREAD = 256, PARALLEL_BLOCK_PER_SM = NUM_THREAD_PER_SM / NUM_THREAD,
            NUM_BLOCK = PARALLEL_BLOCK_PER_SM * NUM_SM;
    constexpr unsigned WARP_NUM = CEIL(NUM_THREAD, WARP_SIZE);
    dot_warp_shuffle_xor_v1<WARP_NUM><<<NUM_BLOCK, NUM_THREAD>>>(N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
}

void call_dot_warp_shuffle_xor_v2(const unsigned N, const float *dev_a, const float *dev_b, float *dev_ret,
                                  float *ret) {
    // init
    cudaMemset(dev_ret, 0, sizeof(float));
    // kernel
    constexpr unsigned NUM_THREAD = 256, PARALLEL_BLOCK_PER_SM = NUM_THREAD_PER_SM / NUM_THREAD,
            NUM_BLOCK = PARALLEL_BLOCK_PER_SM * NUM_SM;
    constexpr unsigned NUM_WARP = CEIL(NUM_THREAD, WARP_SIZE);
    reduce<float, MultipleFunctor, AddFunctor, AtomicAddFunctor, NUM_WARP><<<NUM_BLOCK,NUM_THREAD>>>(
        N, dev_a, dev_b, dev_ret);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // copy output
    cudaMemcpy(ret, dev_ret, sizeof(float), cudaMemcpyDeviceToHost);
}
