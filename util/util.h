//
// Created by hzw on 2025/10/27.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_UTIL_CUH
#define CUDA_TUTORIAL_EXAMPLE_UTIL_CUH


#define WARP_SIZE 32
#define NUM_SM 108
#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FLOAT4(x) (*((float4*)(&x)))
// #define _2D_2_1D(a,i,j,step) ((a)[(i)*(step)+(j)])

template<typename T>
__device__ __host__ __forceinline__ T &_2D_2_1D(T *a, const unsigned i, const unsigned j, const unsigned step) {
    return a[i * step + j];
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

template <typename T>
struct prepare_param {
    T *p;// 不malloc，不需要引用
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
