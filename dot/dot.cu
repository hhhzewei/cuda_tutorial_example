//
// Created by hzw on 2025/10/28.
//
#include <iostream>

#include "call.h"
#include "util/device_memory.h"
#include "util/util.h"

float dot_error(const unsigned N, const float *a, const float *b, const float *ret) {
    float tmp = 0.0f;
    for (int i = 0; i < N; ++i) {
        tmp += a[i] * b[i];
    }
    return fabs(tmp - *ret);
}

int main() {
    constexpr unsigned N = 1 << 20;
    auto initializer_a = [](unsigned i) { return 1.0f; };
    auto initializer_b = [](unsigned i) { return 1.0f; };
    const DeviceMemory<float, true> a_mem(N, initializer_a);
    const DeviceMemory<float, true> b_mem(N, initializer_b);
    const DeviceMemory<float, true> ret_mem(N);
    // call dot cublas
    call_dot_cublas(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "dot cublas error: " << dot_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call dot kernel
    constexpr unsigned threadNum = 256;
    constexpr unsigned blockNum = CEIL(N, threadNum);
    call_dot<blockNum, threadNum>(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p, a_mem.p);
    std::cout << "dot error: " << dot_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call dot shared kernel
    call_dot_shared<blockNum, threadNum>(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "dot shared error: " << dot_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call dot shared external kernel
    call_dot_shared_external<blockNum, threadNum>(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "dot shared external error: " << dot_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call dot warp shuffle down kernel
    call_dot_warp_shuffle_down<blockNum, threadNum>(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "dot shared warp shuffle down error: " << dot_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call dot shared warp shuffle xor v0 kernel
    call_dot_warp_shuffle_xor_v0<blockNum, threadNum>(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "dot shared warp shuffle xor v0 error: " << dot_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    constexpr unsigned PARALLEL_BLOCK_PER_SM = 8;
    call_dot_warp_shuffle_xor_v0<PARALLEL_BLOCK_PER_SM * NUM_SM, threadNum>(
        N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "dot shared warp shuffle xor v0 error: " << dot_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call dot shared warp shuffle xor v1 kernel
    call_dot_warp_shuffle_xor_v1(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "dot shared warp shuffle xor v1 error: " << dot_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call dot shared warp shuffle xor v2 kernel
    call_dot_warp_shuffle_xor_v2(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "dot shared warp shuffle xor v2 error: " << dot_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
}
