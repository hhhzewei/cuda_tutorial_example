//
// Created by hzw on 2025/10/27.
//

#include "check.h"
#include <cstdio>
#include <iostream>

void check_error(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error:%s\n", cudaGetErrorString(err));
    }
}

float add_error(unsigned N, float *a, float *b, float *c) {
    float maxError = 0.0f;
    for (int i = 0; i < N; ++i) {
        maxError = fmax(maxError, fabs(c[i] - a[i] - b[i]));
    }
    return maxError;
}

float dot_error(unsigned N, float *a, float *b, float *ret) {
    float tmp = 0.0f;
    for (int i = 0; i < N; ++i) {
        tmp += a[i] * b[i];
    }
    return fabs(tmp - *ret);
}

float transpose_error(unsigned M, unsigned N, float *input, float *output) {
    float ret = 0.0f;
    for (unsigned i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            ret += fabs(input[i * N + j] - output[j * M + i]);
        }
    }
    return ret;
}

float sgemm_error(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret) {
    float error = 0.0f;
    for (unsigned i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float value = 0.0f;
            for (int k = 0; k < K; ++k) {
                value += a[i * K + k] * b[k * N + j];
            }
            error += fabs(value - ret[i * N + j]);
        }
    }
    return error;
}
