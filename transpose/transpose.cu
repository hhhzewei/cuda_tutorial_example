//
// Created by hzw on 2025/11/2.
//


#include <iostream>
#include "call.h"
#include "util/util.h"

void host_prepare(const unsigned M, const unsigned N,
                  float *&input, float *&output) {
    size_t SIZE = M * N * sizeof(float);
    // host malloc
    input = (float *) malloc(SIZE);
    output = (float *) malloc(SIZE);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            input[i * N + j] = i - j;
        }
    }
}

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
    // host malloc
    float *input, *output;
    float *dev_input, *dev_output;
    cudaStream_t stream_input, stream_output;
    cudaEvent_t kernel_finish;
    // prepare
    host_prepare(M, N, input, output);
    device_prepare<float>({
                              {input, dev_input, M * N, stream_input},
                              {nullptr, dev_output, N * M, stream_output}
                          },
                          kernel_finish);
    // call transpose naive kernel
    call_transpose_naive(M, N, dev_input, dev_output,output);
    std::cout << "transpose naive error: " << transpose_error(M, N, input, output) << std::endl;
    // call transpose sahred kernel
    call_transpose_shared(M, N, dev_input, dev_output,output);
    std::cout << "transpose padding error: " << transpose_error(M, N, input, output) << std::endl;
    // call transpose padding kernel
    call_transpose_padding(M, N, dev_input, dev_output,output);
    std::cout << "transpose padding error: " << transpose_error(M, N, input, output) << std::endl;
    // call transpose swizzle kernel
    call_transpose_swizzle(M, N, dev_input, dev_output,output);
    std::cout << "transpose swizzle error: " << transpose_error(M, N, input, output) << std::endl;
    // call transpose cublas
    call_transpose_cubalas(M, N, dev_input, dev_output,output);
    std::cout << "transpose cublas error: " << transpose_error(M, N, input, output) << std::endl;
    // destroy
    destroy({
                {input, dev_input, stream_input},
                {output, dev_output, stream_output}
            }, kernel_finish);
}
