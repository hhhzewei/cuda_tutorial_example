//
// Created by hzw on 2026/1/1.
//


#include <iostream>

#include "call.h"
#include "util/util.h"

void host_prepare(const unsigned N, float *&x, float *&logits) {
    // host memory malloc
    x = (float *) malloc(N * sizeof(float));
    logits = (float *) malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(i);
    }
}

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
    float *x, *logits;
    float *dev_x, *dev_logits, *dev_maxs, *dev_exp_sums;
    cudaStream_t stream_x, stream_logits, stream_maxs, stream_exp_sums;
    cudaEvent_t kernel_finish;
    // prepare
    host_prepare(N, x, logits);
    constexpr unsigned NUM_THREAD = 256, PARALLEL_BLOCK_PER_SM = NUM_THREAD_PER_SM / NUM_THREAD,
            NUM_BLOCK = PARALLEL_BLOCK_PER_SM * NUM_SM;
    device_prepare<float>({
                              {x, dev_x, N, stream_x},
                              {nullptr, dev_logits, N, stream_logits},
                              {nullptr, dev_maxs, NUM_BLOCK, stream_maxs},
                              {nullptr, dev_exp_sums, NUM_BLOCK, stream_exp_sums}
                          }, kernel_finish);
    // call torch softmax
    call_softmax_torch(N, dev_x, logits);
    std::cout << "torch softmax error: " << softmax_error(N, x, logits) << std::endl;
    // call online softmax
    call_softmax_online<NUM_THREAD, NUM_BLOCK>(N, dev_x, dev_logits, dev_maxs, dev_exp_sums, logits);
    std::cout << "online softmax error: " << softmax_error(N, x, logits) << std::endl;
    // destroy
    destroy({
                {x, dev_x, stream_x},
                {logits, dev_logits, stream_logits},
                {nullptr, dev_maxs, stream_maxs},
                {nullptr, dev_exp_sums, stream_exp_sums}
            }, kernel_finish);
}
