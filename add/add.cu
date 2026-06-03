//
// Created by hzw on 2025/10/28.
//
#include <iostream>
#include "call.h"
#include "util/device_memory.h"
#include "util/util.h"

float add_error(const unsigned N, const float *a, const float *b, const float *c) {
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i) {
        maxError = fmax(maxError, fabs(c[i] - a[i] - b[i]));
    }
    return maxError;
}

int main() {
    constexpr unsigned N = 1 << 20;
    auto initializer_a = [](unsigned) { return 1.0f; };
    auto initializer_b = [](unsigned) { return 2.0f; };
    const DeviceMemory<float, true> a_mem(N, initializer_a);
    const DeviceMemory<float, true> b_mem(N, initializer_b);
    const DeviceMemory<float, true> ret_mem(N);
    // call add cublas
    call_add_cublas(N, a_mem.dev_p, b_mem.dev_p, ret_mem.p, b_mem.p);
    std::cout << "cublas add error: " << add_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call add kernel
    constexpr unsigned threadNum = 256;
    constexpr unsigned blockNum = CEIL(N, threadNum);
    call_add<blockNum, threadNum>(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "add error: " << add_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call add float4
    call_add_float4<blockNum, threadNum>(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "add float4 error: " << add_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call add v1 kernel
    call_add_v1(N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "add v1 error: " << add_error(N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
}
