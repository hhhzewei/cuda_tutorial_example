//
// Created by hzw on 2025/11/2.
//


#include <iostream>

#include "call.h"
#include "check.h"
#include "kernel.h"

int main() {
    unsigned M = 1 << 10, N = 1 << 11;
    unsigned SIZE = M * N * sizeof(float);
    // host malloc
    float *input = (float *) malloc(SIZE), *output = (float *) malloc(SIZE);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            input[i * N + j] = i - j;
        }
    }
    // call transpose naive kernel
    call_transpose_naive(M, N, input, output);
    std::cout << "transpose naive error: " << transpose_error(M, N, input, output) << std::endl;
    // call transpose padding kernel
    call_transpose_padding(M, N, input, output);
    std::cout << "transpose padding error: " << transpose_error(M, N, input, output) << std::endl;
    // call transpose swizzle kernel
    call_transpose_swizzle(M, N, input, output);
    std::cout << "transpose swizzle error: " << transpose_error(M, N, input, output) << std::endl;
    // host free
    free(input);
    free(output);
}
