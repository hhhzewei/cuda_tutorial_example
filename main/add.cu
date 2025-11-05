//
// Created by hzw on 2025/10/28.
//
#include "kernel.h"
#include "check.h"
#include <cstdio>
#include <iostream>

#include "call.h"

int main() {
    constexpr unsigned N = 1 << 20;
    // host memory malloc
    float *a = (float *) malloc(N * sizeof(float)), *b = (float *) malloc(N * sizeof(float)), *ret = (float *) malloc(
        N * sizeof(float));
    for (unsigned i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
    // call add kernel
    constexpr unsigned threadNum = 256;
    constexpr unsigned blockNum = CEIL(N, threadNum);
    call_add<blockNum, threadNum>(N, a, b, ret);
    std::cout << "add error: " << add_error(N, a, b, ret) << std::endl;
    // call add float4
    call_add_float4<blockNum, threadNum>(N, a, b, ret);
    std::cout << "add float4 error: " << add_error(N, a, b, ret) << std::endl;
    // call add cublas
    call_add_cublas(N, a, b, ret);
    // host free
    free(a);
    free(b);
    free(ret);
}
