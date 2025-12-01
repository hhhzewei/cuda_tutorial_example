//
// Created by hzw on 2025/10/28.
//
#include <iostream>
#include "call.h"
#include "util/util.h"

void host_prepare(const unsigned N,
                  float *&a, float *&b, float *&ret) {
    // host memory malloc
    a = (float *) malloc(N * sizeof(float));
    b = (float *) malloc(N * sizeof(float));
    ret = (float *) malloc(N * sizeof(float));
    for (unsigned i = 0; i < N; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }
}

float add_error(const unsigned N, const float *a, const float *b, const float *c) {
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i) {
        maxError = fmax(maxError, fabs(c[i] - a[i] - b[i]));
    }
    return maxError;
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
                              {nullptr, dev_ret, N, stream_ret}
                          }, kernel_finish);
    // call add kernel
    constexpr unsigned threadNum = 256;
    constexpr unsigned blockNum = CEIL(N, threadNum);
    call_add<blockNum, threadNum>(N, dev_a, dev_b, dev_ret, ret);
    std::cout << "add error: " << add_error(N, a, b, ret) << std::endl;
    // call add float4
    call_add_float4<blockNum, threadNum>(N, dev_a, dev_b, dev_ret, ret);
    std::cout << "add float4 error: " << add_error(N, a, b, ret) << std::endl;
    // call add cublas
    call_add_cublas(N, dev_a, dev_b, ret, b);
    std::cout << "cublas add error: " << add_error(N, a, b, ret) << std::endl;
    // destroy
    destroy({
                {a, dev_a, stream_a},
                {b, dev_b, stream_b},
                {ret, dev_ret, stream_ret}
            }, kernel_finish);
}
