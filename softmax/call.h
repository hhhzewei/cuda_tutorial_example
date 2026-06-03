//
// Created by hzw on 2026/1/2.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CALL_H
#define CUDA_TUTORIAL_EXAMPLE_CALL_H
#include "util/device_memory.h"

void call_softmax_online(unsigned N, const DeviceMemory<float, true> &x_mem,
                         const DeviceMemory<float, true> &logits_mem);

void call_softmax_naive(unsigned N, const DeviceMemory<float, true> &x_mem,
                        const DeviceMemory<float, true> &logits_mem);

void call_softmax_torch(unsigned N, float *dev_x, float *logits);

#endif //CUDA_TUTORIAL_EXAMPLE_CALL_H
