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

struct NoInit{};

template<typename T, bool need_host, typename Initializer = NoInit>
struct DeviceMemory {
    T *p = nullptr; // 不malloc，不需要引用
    T *dev_p;
    unsigned size;
    cudaStream_t stream;

    explicit DeviceMemory(const unsigned size,Initializer initializer) : size(size), stream() {
        cudaStreamCreate(&stream);
        cudaMallocAsync(&dev_p, size * sizeof(T), stream);
        if constexpr (need_host) {
            p = static_cast<T *>(malloc(size * sizeof(T)));
            if constexpr (!std::is_same_v<NoInit, Initializer>) {
                for (unsigned i = 0; i < size; ++i) {
                    p[i] = initializer(i);
                }
                cudaMemcpyAsync(dev_p, p, size * sizeof(T), cudaMemcpyHostToDevice, stream);
            }
        }
    }

    DeviceMemory(const DeviceMemory &x) = delete;

    ~DeviceMemory() {
        cudaFreeAsync(dev_p, stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        if constexpr (need_host) {
            free(p);
        }
    }
};

#endif //CUDA_TUTORIAL_EXAMPLE_UTIL_CUH
