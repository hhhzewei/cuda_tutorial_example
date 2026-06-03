//
// Created by hzw on 2026/1/1.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_REDUCE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_REDUCE_KERNEL_H
#include "util/util.h"

template<typename T,
    typename CalFunc, typename ReduceFunc, typename AtomicFunc,
    unsigned NUM_WARP>
__global__ void reduce(const unsigned N, const T *a, T *b,
                       CalFunc cal_func = CalFunc{}, ReduceFunc reduce_func = ReduceFunc{},
                       AtomicFunc atomic_func = AtomicFunc{}) {
    __shared__ float s_mem[NUM_WARP];
    constexpr T INIT_VALUE = ReduceFunc::init();
    const unsigned threadIdxGlobal = blockIdx.x * blockDim.x + threadIdx.x, NUM_THREAD = gridDim.x * blockDim.x,
            warpIdx = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if (threadIdxGlobal == 0) {
        *b = INIT_VALUE;
    }
    T result = INIT_VALUE;
    for (unsigned i = threadIdxGlobal; i < N; i += NUM_THREAD) {
        result = reduce_func(result, cal_func(a[i]));
    }
    __syncwarp();
    shuffle_xor_reduce<float, ReduceFunc, WARP_SIZE>(result);
    if (lane == 0) {
        s_mem[warpIdx] = result;
    }
    __syncthreads();
    result = lane < NUM_WARP ? s_mem[lane] : INIT_VALUE;
    __syncwarp();
    shuffle_xor_reduce<float, ReduceFunc, NUM_WARP>(result);
    if (warpIdx == 0 && lane == 0) {
        atomic_func(b, result);
    }
}

template<typename T,
    typename CalFunc, typename ReduceFunc, typename AtomicFunc,
    unsigned NUM_WARP>
__global__ void reduce(const unsigned N, const T *a, const T *b, T *c,
                       CalFunc cal_func = CalFunc{}, ReduceFunc reduce_func = ReduceFunc{},
                       AtomicFunc atomic_func = AtomicFunc{}) {
    __shared__ float s_mem[NUM_WARP];
    constexpr T INIT_VALUE = ReduceFunc::init();
    const unsigned threadIdxGlobal = blockIdx.x * blockDim.x + threadIdx.x, NUM_THREAD = gridDim.x * blockDim.x,
            warpIdx = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    if (threadIdxGlobal == 0) {
        *c = INIT_VALUE;
    }
    T result = INIT_VALUE;
    for (unsigned i = threadIdxGlobal; i < N; i += NUM_THREAD) {
        result = reduce_func(result, cal_func(a[i], b[i]));
    }
    __syncwarp();
    shuffle_xor_reduce<float, ReduceFunc, WARP_SIZE>(result);
    if (lane == 0) {
        s_mem[warpIdx] = result;
    }
    __syncthreads();
    result = lane < NUM_WARP ? s_mem[lane] : INIT_VALUE;
    __syncwarp();
    shuffle_xor_reduce<float, ReduceFunc, NUM_WARP>(result);
    if (warpIdx == 0 && lane == 0) {
        atomic_func(c, result);
    }
}

#endif //CUDA_TUTORIAL_EXAMPLE_REDUCE_KERNEL_H
