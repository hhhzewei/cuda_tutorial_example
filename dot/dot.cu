//
// Created by hzw on 2025/10/28.
//
#include <iostream>

#include "call.h"
#include "util/util.h"

void host_prepare(const unsigned N, float *&a, float *&b, float *&ret) {
    // host memory malloc
    a = (float *) malloc(N * sizeof(float));
    b = (float *) malloc(N * sizeof(float));
    ret = (float *) malloc(sizeof(float));
    for (int i = 0; i < N; ++i) {
        a[i] = b[i] = 1.0f;
    }
}

float dot_error(const unsigned N, const float *a, const float *b, const float *ret) {
    float tmp = 0.0f;
    for (int i = 0; i < N; ++i) {
        tmp += a[i] * b[i];
    }
    return fabs(tmp - *ret);
}

int main() {
    constexpr unsigned N = 1 << 20;
    float *a, *b, *ret;
    float *dev_a, *dev_b, *dev_ret;
    cudaStream_t stream_a, stream_b, stream_ret;
    cudaEvent_t kernel_finish;
    // prepare
    host_prepare(N, a, b, ret);
    device_prepare<float>({
                              {a, dev_a, N, stream_a},
                              {b, dev_b, N, stream_b},
                              {nullptr, dev_ret, 1, stream_ret}
                          }, kernel_finish);
    // call dot cublas
    call_dot_cublas(N, dev_a, dev_b, dev_ret, ret);
    std::cout << "dot cublas error: " << dot_error(N, a, b, ret) << std::endl;
    // call dot kernel
    constexpr unsigned threadNum = 256;
    constexpr unsigned blockNum = CEIL(N, threadNum);
    call_dot<blockNum, threadNum>(N, dev_a, dev_b, dev_ret, ret, a);
    std::cout << "dot error: " << dot_error(N, a, b, ret) << std::endl;
    // call dot shared kernel
    call_dot_shared<blockNum, threadNum>(N, dev_a, dev_b, dev_ret, ret);
    std::cout << "dot shared error: " << dot_error(N, a, b, ret) << std::endl;
    // call dot shared external kernel
    call_dot_shared_external<blockNum, threadNum>(N, dev_a, dev_b, dev_ret, ret);
    std::cout << "dot shared external error: " << dot_error(N, a, b, ret) << std::endl;
    // call dot warp shuffle down kernel
    call_dot_warp_shuffle_down<blockNum, threadNum>(N, dev_a, dev_b, dev_ret, ret);
    std::cout << "dot shared warp shuffle down error: " << dot_error(N, a, b, ret) << std::endl;
    // call dot shared warp shuffle xor v0 kernel
    call_dot_warp_shuffle_xor_v0<blockNum, threadNum>(N, dev_a, dev_b, dev_ret, ret);
    std::cout << "dot shared warp shuffle xor v0 error: " << dot_error(N, a, b, ret) << std::endl;
    constexpr unsigned PARALLEL_BLOCK_PER_SM = 8;
    call_dot_warp_shuffle_xor_v0<PARALLEL_BLOCK_PER_SM * NUM_SM, threadNum>(N, dev_a, dev_b, dev_ret, ret);
    std::cout << "dot shared warp shuffle xor v0 error: " << dot_error(N, a, b, ret) << std::endl;
    // call dot shared warp shuffle xor v1 kernel
    call_dot_warp_shuffle_xor_v1<PARALLEL_BLOCK_PER_SM * NUM_SM, threadNum>(N, dev_a, dev_b, dev_ret, ret);
    std::cout << "dot shared warp shuffle xor v1 error: " << dot_error(N, a, b, ret) << std::endl;
    // destroy
    destroy({
                {a, dev_a, stream_a},
                {b, dev_b, stream_b},
                {ret, dev_ret, stream_ret}
            }, kernel_finish);
}
