//
// Created by hzw on 2025/10/28.
//
#include "kernel.h"
#include "check.h"
#include <cstdio>

#include "call.h"

int main(void) {
    constexpr unsigned N = 1 << 20;
    // host memory malloc
    float *a = (float *) malloc(N * sizeof(float)), *b = (float *) malloc(N * sizeof(float)), *ret = (float *) malloc(
        N * sizeof(float));
    for (unsigned i = 0; i < N; ++i) {
        a[i]=1.0f;
        b[i]=2.0f;
    }
    // call add kernel
    constexpr unsigned threadNum = 256;
    constexpr unsigned blockNum = CEIL(N, threadNum);
    call_add<blockNum, threadNum>(N, a, b, ret);
    printf("add error: %f", add_error(N, a, b, ret));
    // call add float4
    call_add_float4<blockNum, threadNum>(N, a, b, ret);
    printf("add float4 error: %f", add_error(N, a, b, ret));

    // host free
    free(a);
    free(b);
    free(ret);
}
