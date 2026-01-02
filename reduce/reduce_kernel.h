//
// Created by hzw on 2026/1/1.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_REDUCE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_REDUCE_KERNEL_H
#include "../util/util.h"

template<typename T,
    template<typename> class CalFunc, template<typename> class ReduceFunc, template<typename> class AtomicFunc,
    unsigned NUM_WARP>
__global__ void reduce(const unsigned N, const T *a, const T *b, T *c) {
    __shared__ float s_mem[NUM_WARP];
    CalFunc<T> cal_func{};
    ReduceFunc<T> reduce_func{};
    AtomicFunc<T> atomic_func{};
    const unsigned threadIdxGlobal = blockIdx.x * blockDim.x + threadIdx.x, NUM_THREAD = gridDim.x * blockDim.x,
            warpIdx = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    T result{};
    for (unsigned i = threadIdxGlobal; i < N; i += NUM_THREAD) {
        result = reduce_func(result, cal_func(a[i], b[i]));
    }
    __syncwarp();
    shuffle_xor_reduce<float, ReduceFunc, WARP_SIZE>(result);
    if (lane == 0) {
        s_mem[warpIdx] = result;
    }
    __syncthreads();
    result = s_mem[lane];
    __syncwarp();
    shuffle_xor_reduce<float, ReduceFunc, NUM_WARP>(result);
    if (warpIdx == 0 && lane == 0) {
        atomic_func(c, result);
    }
}

#endif //CUDA_TUTORIAL_EXAMPLE_REDUCE_KERNEL_H
