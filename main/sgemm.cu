//
// Created by hzw on 2025/11/2.
//


#include <iostream>

#include "call.h"
#include "check.h"

int main() {
    constexpr unsigned M = 1 << 10, N = 1 << 11, K = 1 << 12;
    // host malloc
    float *a = (float *) malloc(M * K * sizeof(float)),
            *b = (float *) malloc(K * N * sizeof(float)),
            *ret = (float *) malloc(M * N * sizeof(float));
    for (unsigned i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            a[i * K + k] = 1.0f;
        }
    }
    for (unsigned k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            b[k * N + j] = 1.0f;
        }
    }
    // call sgemm naive kernel
    call_sgemm_naive(M, K, N, a, b, ret);
    std::cout << "sgemm naive error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm block tile kernel
    call_sgemm_block_tile(M, K, N, a, b, ret);
    std::cout << "sgemm block tile error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm thread tile kernel
    call_sgemm_thread_tile_v0(M, K, N, a, b, ret);
    std::cout << "sgemm thread tile v0 error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm thread tile v1 kernel
    call_sgemm_thread_tile_v1(M, K, N, a, b, ret);
    std::cout << "sgemm thread tile v1 result:" << *ret << ", error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // // call cublast sgemm
    call_sgemm_cublas(M, K, N, a, b, ret);
    // host free
    free(a);
    free(b);
    free(ret);
}
