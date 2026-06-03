//
// Created by hzw on 2025/10/27.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_UTIL_CUH
#define CUDA_TUTORIAL_EXAMPLE_UTIL_CUH

// A100
#define NUM_SM 108
#define NUM_THREAD_PER_SM 2048
#define MAX_NUM_THREAD_PER_BLOCK 1024

#define WARP_SIZE 32
#define NUM_SM 108
#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FLOAT4(x) (*((float4*)(&x)))

#include <cfloat>
// #define _2D_2_1D(a,i,j,step) ((a)[(i)*(step)+(j)])

template<typename T>
__device__ __host__ __forceinline__ T &_2D_2_1D(T *a, const unsigned i, const unsigned j, const unsigned step) {
    return a[i * step + j];
}

template<typename T, typename ReduceFunc, unsigned NUM>
__device__ __forceinline__ void shuffle_xor_reduce(T &value, ReduceFunc reduce_func = {}) {
#pragma unroll
    for (unsigned offset = NUM >> 1; offset > 0; offset >>= 1) {
        value = reduce_func(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
}

template<typename T, typename ReduceFunc, unsigned NUM>
__device__ __forceinline__ void shuffle_down_reduce(T &value, ReduceFunc reduce_func = {}) {
#pragma unroll
    for (unsigned offset = NUM >> 1; offset > 0; offset >>= 1) {
        value = reduce_func(value, __shfl_down_sync(0xffffffff, value, offset));
    }
}

namespace my_math {
    __device__ __host__ __forceinline__ float fmax(float a, float b) {
        return ::fmaxf(a, b);
    }

    __device__ __host__ __forceinline__ float exp(float x) {
        return ::expf(x);
    }

    template<typename T>
    __device__ __host__ __forceinline__ constexpr T max_value() {
        return T{};
    }

    template<>
    __device__ __host__ __forceinline__ constexpr float max_value<float>() {
        return FLT_MAX;
    }

    template<typename T>
    __device__ __host__ __forceinline__ constexpr T min_value() {
        return T{};
    }

    template<>
    __device__ __host__ __forceinline__ constexpr float min_value<float>() {
        return -FLT_MAX;
    }
}

#define CHECK_ERROR cudaError_t err=cudaGetLastError();    \
if(err != cudaSuccess){    \
printf("Error:%s\n",cudaGetErrorString(err));    \
}    \
else{    \
printf("CudaSuccess\n"); \
}

void check_error(cudaError_t err);

template<typename T>
float gemm_error(const unsigned M, const unsigned K, const unsigned N, const T *a, const T *b,
                 const T *ret) {
    float error = 0.0f;
    for (unsigned i = 0; i < M; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            float value = 0;
            for (unsigned k = 0; k < K; ++k) {
                value += static_cast<float>(a[i * K + k]) * static_cast<float>(b[k * N + j]);
            }
            error = fmaxf(error, fabs(value - static_cast<float>(ret[i * N + j])));
        }
    }
    return error;
}

int getSmCount(int device_id);

#endif //CUDA_TUTORIAL_EXAMPLE_UTIL_CUH
