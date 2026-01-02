//
// Created by hzw on 2026/1/1.

#ifndef CUDA_TUTORIAL_EXAMPLE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_KERNEL_H

#include "../util/util.h"

template<typename T, unsigned NUM>
__device__ __forceinline__ void shuffle_reduce_softmax(T &max, T &exp_sum) {
    for (unsigned offset = NUM >> 1; offset > 0; offset >>= 1) {
        T new_max = my_math::fmax(max, __shfl_xor_sync(0xffffffff, max, offset));
        exp_sum *= my_math::exp(max - new_max);
        max = new_max;
        exp_sum += __shfl_xor_sync(0xffffffff, exp_sum, offset);
    }
}

/**
 *  online softmax
 *
 * @tparam T 数据类型
 * @tparam NUM_WARP warp数
 * @param x 输入向量指针
 * @param logits 输出向量指针
 * @param N 向量维度
 * @param maxs 按block reduce的最大值
 * @param exp_sums 按block reduce 的指数和
 */
template<typename T, unsigned NUM_WARP>
__global__ void softmax_online_1(const unsigned N, T *x, T *maxs, T *exp_sums) {
    __shared__ T s_exp_sum[NUM_WARP];
    __shared__ T s_max[NUM_WARP];
    constexpr T MIN_VALUE = -my_math::max_value<T>();
    const unsigned threadIdxGlobal = threadIdx.x + blockIdx.x * blockDim.x, NUM_THREAD = gridDim.x * blockDim.x,
            warpIdx = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    T max = MIN_VALUE, exp_sum = 0;
    for (unsigned i = threadIdxGlobal; i < N; i += NUM_THREAD) {
        T new_max = my_math::fmax(max, x[i]);
        exp_sum *= my_math::exp(max - new_max);
        exp_sum += my_math::exp(x[i] - new_max);
        max = new_max;
    }
    __syncwarp();
    shuffle_reduce_softmax<T,WARP_SIZE>(max, exp_sum);
    if (lane == 0) {
        s_max[warpIdx] = max;
        s_exp_sum[warpIdx] = exp_sum;
    }
    __syncthreads();
    exp_sum = lane < NUM_WARP ? s_exp_sum[lane] : 0;
    max = lane < NUM_WARP ? s_max[lane] : MIN_VALUE;
    shuffle_reduce_softmax<T, NUM_WARP>(max, exp_sum);
    if (threadIdx.x == 0) {
        maxs[blockIdx.x] = max;
        exp_sums[blockIdx.x] = exp_sum;
    }
}

template<typename T, unsigned NUM_WARP>
__global__ void softmax_online_2(const unsigned N, T *x, T *logits, const unsigned N_block, T *maxs, T *exp_sums) {
    __shared__ T s_max[NUM_WARP], s_exp_sum[NUM_WARP];
    const unsigned threadIdxGlobal = threadIdx.x + blockIdx.x * blockDim.x, NUM_THREAD = gridDim.x * blockDim.x,
            warpIdx = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
    // reduce
    T max = -my_math::max_value<T>(), exp_sum = 0;
    // 每个block都要完成reduce
    for (unsigned i = threadIdx.x; i < N_block; i += blockDim.x) {
        T tmp_max = maxs[i];
        T new_max = my_math::fmax(max, tmp_max);
        exp_sum *= my_math::exp(max - new_max);
        max = new_max;
        exp_sum += exp_sums[i] * my_math::exp(tmp_max - max);
    }
    __syncwarp();
    shuffle_reduce_softmax<T,WARP_SIZE>(max, exp_sum);
    if (lane == 0) {
        s_max[warpIdx] = max;
        s_exp_sum[warpIdx] = exp_sum;
    }
    __syncthreads();
    max = lane < NUM_WARP ? s_max[lane] : -my_math::max_value<float>();
    exp_sum = lane < NUM_WARP ? s_exp_sum[lane] : 0;
    shuffle_reduce_softmax<T, NUM_WARP>(max, exp_sum);
    // broadcast
    max = __shfl_sync(0xffffffff, max, 0);
    exp_sum = __shfl_sync(0xffffffff, exp_sum, 0);
    // 计算结果
    for (unsigned i = threadIdxGlobal; i < N; i += NUM_THREAD) {
        logits[i] = my_math::exp(x[i] - max) / exp_sum;
    }
}
#endif //CUDA_TUTORIAL_EXAMPLE_KERNEL_H
