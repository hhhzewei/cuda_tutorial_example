//
// Created by hzw on 2025/11/2.
//


#include <iostream>
#include "call.h"
#include "util/util.h"

float transpose_error(const unsigned M, const unsigned N, const float *input, const float *output) {
    float ret = 0.0f;
    for (unsigned i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            ret += fabs(input[i * N + j] - output[j * M + i]);
        }
    }
    return ret;
}

int main() {
    constexpr unsigned M = 1 << 10, N = 1 << 11;
    auto initializer_input = [](unsigned i) { return i; };
    const DeviceMemory<float, true> input_mem(M * N, initializer_input);
    const DeviceMemory<float, true> output_mem(M * N);
    // call transpose naive kernel
    call_transpose_naive(M, N, input_mem.dev_p, output_mem.dev_p, output_mem.p);
    std::cout << "transpose naive error: " << transpose_error(M, N, input_mem.p, output_mem.p) << std::endl;
    // call transpose sahred kernel
    call_transpose_shared(M, N, input_mem.dev_p, output_mem.dev_p, output_mem.p);
    std::cout << "transpose padding error: " << transpose_error(M, N, input_mem.p, output_mem.p) << std::endl;
    // call transpose padding kernel
    call_transpose_padding(M, N, input_mem.dev_p, output_mem.dev_p, output_mem.p);
    std::cout << "transpose padding error: " << transpose_error(M, N, input_mem.p, output_mem.p) << std::endl;
    // call transpose swizzle kernel
    call_transpose_swizzle(M, N, input_mem.dev_p, output_mem.dev_p, output_mem.p);
    std::cout << "transpose swizzle error: " << transpose_error(M, N, input_mem.p, output_mem.p) << std::endl;
    // call transpose cublas
    call_transpose_cubalas(M, N, input_mem.dev_p, output_mem.dev_p, output_mem.p);
    std::cout << "transpose cublas error: " << transpose_error(M, N, input_mem.p, output_mem.p) << std::endl;
}
