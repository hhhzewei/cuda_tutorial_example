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
    auto initializer_x = [](unsigned i) { return i; };
    const DeviceMemory<float, true, decltype(initializer_x)> x_mem(N, initializer_x);
    const DeviceMemory<float, true> logits_mem(N, NoInit{});
    // prepare
    constexpr unsigned NUM_THREAD = 256, PARALLEL_BLOCK_PER_SM = NUM_THREAD_PER_SM / NUM_THREAD,
            NUM_BLOCK = PARALLEL_BLOCK_PER_SM * NUM_SM;
    const DeviceMemory<float, false> maxs_mem(NUM_BLOCK, NoInit{});
    const DeviceMemory<float, false> exp_sums_mem(NUM_BLOCK, NoInit{});
    // call torch softmax
    call_softmax_torch(N, x_mem.dev_p, logits_mem.p);
    std::cout << "torch softmax error: " << softmax_error(N, x_mem.p, logits_mem.p) << std::endl;
    // call online softmax
    call_softmax_online<NUM_THREAD, NUM_BLOCK>(N, x_mem.dev_p, logits_mem.dev_p, maxs_mem.dev_p, exp_sums_mem.dev_p,
                                               logits_mem.p);
    std::cout << "online softmax error: " << softmax_error(N, x_mem.p, logits_mem.p) << std::endl;
}
