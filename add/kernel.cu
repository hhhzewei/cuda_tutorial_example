//
// Created by hzw on 2025/10/25.
//
// Kernel function to add the elements of two arrays
#include "kernel.h"
#include "util/util.h"

__global__ void add(const unsigned N, const float *a, const float *b, float *ret) {
    const unsigned strip = gridDim.x * blockDim.x;
    for (unsigned i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += strip)
        ret[i] = a[i] + b[i];
}

__global__ void add_float4(const unsigned N, const float *a, const float *b, float *ret) {
    for (unsigned int i = (threadIdx.x + blockDim.x * blockIdx.x) * 4; i < N; i += blockDim.x * gridDim.x * 4) {
        const float4 tmp_a = FLOAT4(a[i]), tmp_b = FLOAT4(b[i]);
        float4 tmp_ret;
        tmp_ret.x = tmp_a.x + tmp_b.x;
        tmp_ret.y = tmp_a.y + tmp_b.y;
        tmp_ret.z = tmp_a.z + tmp_b.z;
        tmp_ret.w = tmp_a.w + tmp_b.w;
        FLOAT4(ret[i]) = tmp_ret;
    }
}
