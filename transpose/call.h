//
// Created by hzw on 2025/11/4.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CALL_H
#define CUDA_TUTORIAL_EXAMPLE_CALL_H

void prepare_transpose(unsigned M, unsigned N, const float *input, float *&dev_input, float *&dev_output);

void call_transpose_naive(unsigned M, unsigned N, const float *input, float *output);

void call_transpose_shared(unsigned M, unsigned N, const float *input, float *output);

void call_transpose_padding(unsigned M, unsigned N, float *input, float *output);

void call_transpose_swizzle(unsigned M, unsigned N, float *input, float *output);

void call_transpose_cubalas(unsigned M, unsigned N, float *input, float *output);

#endif //CUDA_TUTORIAL_EXAMPLE_CALL_H
