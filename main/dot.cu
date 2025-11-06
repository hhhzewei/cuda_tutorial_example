//
// Created by hzw on 2025/10/28.
//
#include <cstdio>
#include <iostream>

#include "call.h"
#include "kernel.h"
#include "check.h"

int main(void) {
    constexpr unsigned N = 1 << 20;
    // host memory
    float *a = (float *) malloc(N * sizeof(float)), *b = (float *) malloc(N * sizeof(float)), *ret = (float *)
            malloc(sizeof(float));
    for (int i = 0; i < N; ++i) {
        a[i] = b[i] = 1.0f;
    }
    // call dot kernel
    constexpr unsigned threadNum = 256;
    constexpr unsigned blockNum = CEIL(N, threadNum);
    call_dot<blockNum, threadNum>(N, a, b, ret);
    std::cout << "dot error: " << dot_error(N, a, b, ret) << std::endl;
    // call dot shared kernel
    call_dot_shared<blockNum, threadNum>(N, a, b, ret);
    std::cout << "dot shared error: " << dot_error(N, a, b, ret) << std::endl;
    // call dot shared external kernel
    call_dot_shared_external<blockNum, threadNum>(N, a, b, ret);
    std::cout << "dot shared external error: " << dot_error(N, a, b, ret) << std::endl;
    // call dot shared warp shuffle kernel
    call_dot_warp_shuffle<blockNum, threadNum>(N, a, b, ret);
    std::cout << "dot shared warp shuffle error: " << *ret << " " << dot_error(N, a, b, ret) << std::endl;
    // call dot cublas
    call_dot_cublas(N, a, b, ret);
    // host free
    free(a);
    free(b);
    free(ret);
}
