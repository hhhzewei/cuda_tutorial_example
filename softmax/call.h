//
// Created by hzw on 2026/1/2.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CALL_H
#define CUDA_TUTORIAL_EXAMPLE_CALL_H
#include "kernel.h"
#include "../util/util.h"

template<unsigned NUM_THREAD, unsigned NUM_BLOCK>
void call_softmax_online(unsigned N, float *dev_x, float *dev_logits, float *dev_maxs, float *dev_exp_sums,
                         float *logits) {
    // kernel
    constexpr unsigned NUM_WARP = CEIL(NUM_THREAD, WARP_SIZE);
    softmax_online_1<float, NUM_WARP><<<NUM_BLOCK,NUM_THREAD>>>(N, dev_x, dev_maxs, dev_exp_sums);
    softmax_online_2<float, NUM_WARP><<<NUM_BLOCK, NUM_THREAD>>>(N, dev_x, dev_logits, NUM_BLOCK, dev_maxs, dev_exp_sums);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // 写回
    cudaMemcpy(logits, dev_logits, N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_softmax_torch(unsigned N, float *dev_x, float *logits);

#endif //CUDA_TUTORIAL_EXAMPLE_CALL_H
