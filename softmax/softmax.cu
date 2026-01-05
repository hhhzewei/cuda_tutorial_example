//
// Created by hzw on 2026/1/1.
//


#include <iostream>

#include "call.h"
#include "util/util.h"

float softmax_error(const unsigned N, const float *x, const float *logits) {
    float max = -FLT_MAX;
    for (unsigned i = 0; i < N; ++i)max = fmaxf(max, x[i]);
    float exp_sum = 0.0;
    for (unsigned i = 0; i < N; ++i)exp_sum += expf(x[i] - max);
    float error = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        error += fabs(logits[i] - expf(x[i] - max) / exp_sum);
    }
    return error;
}

int main() {
    constexpr unsigned N = 1 << 20;
    auto initializer_x = [](unsigned i) { return static_cast<float>(i) - (N >> 1); };
    const DeviceMemory<float, true> x_mem(N, initializer_x);
    const DeviceMemory<float, true> logits_mem(N, NoInit{});
    // call torch softmax
    call_softmax_torch(N, x_mem.dev_p, logits_mem.p);
    std::cout << "torch softmax error: " << softmax_error(N, x_mem.p, logits_mem.p) << std::endl;
    // call online softmax
    call_softmax_online(N, x_mem, logits_mem);
    std::cout << "online softmax error: " << softmax_error(N, x_mem.p, logits_mem.p) << std::endl;
    // call naive softmax
    call_softmax_naive(N, x_mem, logits_mem);
    std::cout << "naive softmax error: " << softmax_error(N, x_mem.p, logits_mem.p) << std::endl;
}
