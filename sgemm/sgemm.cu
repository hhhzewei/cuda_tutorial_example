//
// Created by hzw on 2025/11/2.
//


#include <iostream>
#include "call.h"
#include "util/util.h"

void host_prepare(const unsigned M, const unsigned N, const unsigned K,
                  float *&a, float *&b, float *&ret) {
    // host memory
    a = (float *) malloc(M * K * sizeof(float));
    b = (float *) malloc(K * N * sizeof(float));
    ret = (float *) malloc(M * N * sizeof(float));
    // data init
    for (unsigned i = 0; i < M; ++i) {
        for (unsigned k = 0; k < K; ++k) {
            a[i * K + k] = 0.1f;
        }
    }
    for (unsigned k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            b[k * N + j] = 0.2f;
        }
    }
}

float sgemm_error(const unsigned M, const unsigned K, const unsigned N, const float *a, const float *b,
                  const float *ret) {
    float error = 0.0f;
    for (unsigned i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float value = 0.0f;
            for (int k = 0; k < K; ++k) {
                value += a[i * K + k] * b[k * N + j];
            }
            error = fmaxf(error, fabs(value - ret[i * N + j]));
        }
    }
    return error;
}

int main() {
    constexpr unsigned M = 1 << 11, N = 1 << 11, K = 1 << 12;
    float *a, *b, *ret;
    float *dev_a, *dev_b, *dev_ret;
    cudaStream_t stream_a, stream_b, stream_ret;
    cudaEvent_t kernel_finish;
    // prepare
    host_prepare(M, N, K, a, b, ret);
    device_prepare<float>({
                              {a, dev_a, M * K, stream_a},
                              {b, dev_b, K * N, stream_b},
                              {nullptr, dev_ret, M * N, stream_ret}
                          }, kernel_finish);
    // call cublas sgemm
    call_sgemm_cublas(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "call sgemm cublas kernel: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call cutlass sgemm colab装cutlass太麻烦了
    // call sgemm naive kernel
    call_sgemm_naive(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "sgemm naive error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm block tile kernel
    call_sgemm_block_tile(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "sgemm block tile error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm thread tile kernel
    call_sgemm_thread_tile_v0(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "sgemm thread tile v0 error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm thread tile v1 kernel太慢永久封印
    // call sgemm thread tile v2 kernel
    call_sgemm_thread_tile_v2(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "call sgemm thread tile v2 padding error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm thread tile v3 kernel
    call_sgemm_thread_tile_v3(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "call sgemm thread tile v3 error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm thread tile v4 kernel
    call_sgemm_thread_tile_v4(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "call sgemm thread tile v4 error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm thread tile v5 kernel
    call_sgemm_thread_tile_v5(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "call sgemm thread tile v5 error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm thread tile v6 kernel
    call_sgemm_thread_tile_v6(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "call sgemm thread tile v6 error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm thread tile v7 kernel
    call_sgemm_thread_tile_v7(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "call sgemm thread tile v7 error: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm tensor core v0 kernel 性能不行，一个block一个warp也太蠢了
    // call sgemm tensor core v1 kernel
    call_sgemm_tensor_core_v1(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "call sgemm tensor core v1 kernel: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // call sgemm tensor core v2 kernel
    call_sgemm_tensor_core_v2(M, K, N, dev_a, dev_b, dev_ret, ret);
    std::cout << "call sgemm tensor core v2 kernel: " << sgemm_error(M, K, N, a, b, ret) << std::endl;
    // host free
    destroy({
                {a, dev_a, stream_a},
                {b, dev_b, stream_b},
                {ret, dev_ret, stream_ret}
            },
            kernel_finish);
}
