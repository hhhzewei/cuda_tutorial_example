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

template<typename T, template<typename>class ReduceFunc, unsigned NUM>
__device__ __forceinline__ void shuffle_xor_reduce(T &value) {
    ReduceFunc<T> reduce_func{};
#pragma unroll
    for (unsigned offset = NUM >> 1; offset > 0; offset >>= 1) {
        value = reduce_func(value, __shfl_xor_sync(0xffffffff, value, offset));
    }
}

template<typename T, template<typename>class ReduceOp, unsigned NUM>
__device__ __forceinline__ void shuffle_down_reduce(T &value) {
    ReduceOp<T> reduce_op{};
#pragma unroll
    for (unsigned offset = NUM >> 1; offset > 0; offset >>= 1) {
        value = reduce_op(value, __shfl_down_sync(0xffffffff, value, offset));
    }
}

namespace my_math {
    __device__ __forceinline__ float fmax(float a, float b) {
        return ::fmaxf(a, b);
    }

    __device__ __forceinline__ float exp(float x) {
        return ::expf(x);
    }

    template<typename T>
    __device__ __forceinline__ constexpr T max_value() {
        return T{};
    }

    template<>
    __device__ __forceinline__ constexpr float max_value<float>() {
        return FLT_MAX;
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

void batch_free(std::initializer_list<void *> ptr_list);

void batch_cuda_free(std::initializer_list<void *> ptr_list);

template<typename T>
struct prepare_param {
    T *p; // 不malloc，不需要引用
    T *&dev_p;
    unsigned size;
    cudaStream_t &stream;
};

template<typename T>
void device_prepare(std::initializer_list<prepare_param<T> > param_list, cudaEvent_t &kernel_finish) {
    for (auto param: param_list) {
        // create stream
        cudaStreamCreate(&param.stream);
        // cuda malloc
        cudaMallocAsync(&param.dev_p, param.size * sizeof(T), param.stream);
        // cuda memcpy
        if (param.p) {
            // p为null表示result矩阵，不需要传输数据
            cudaMemcpyAsync(param.dev_p, param.p, param.size * sizeof(T), cudaMemcpyHostToDevice, param.stream);
        }
    }
    cudaEventCreate(&kernel_finish);
}

struct destroy_param {
    void *p;
    void *dev_p;
    cudaStream_t &stream;
};

void destroy(std::initializer_list<destroy_param> param_list, const cudaEvent_t &kernel_finish);
#endif //CUDA_TUTORIAL_EXAMPLE_UTIL_CUH
