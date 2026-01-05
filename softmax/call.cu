//
// Created by hzw on 2026/1/1.
//
#include <torch/torch.h>
#include "call.h"

#include "element_wise/element_wise_kernel.h"
#include "reduce/reduce_kernel.h"
#include "util/functor.h"

void call_softmax_torch(unsigned N, float *dev_x, float *logits) {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(dev_x, {1, N}, options);
    // 执行运算
    torch::Tensor result = torch::softmax(input, 1);
    // 拷贝结果回主机
    cudaMemcpy(logits, result.data_ptr<float>(), N * sizeof(float), cudaMemcpyDeviceToHost);
}

void call_softmax_online(const unsigned N, const DeviceMemory<float, true> &x_mem,
                         const DeviceMemory<float, true> &logits_mem) {
    constexpr unsigned NUM_THREAD = 256, PARALLEL_BLOCK_PER_SM = NUM_THREAD_PER_SM / NUM_THREAD,
            NUM_BLOCK = PARALLEL_BLOCK_PER_SM * NUM_SM;
    const DeviceMemory<float, false> maxs_mem(NUM_BLOCK, NoInit{});
    const DeviceMemory<float, false> exp_sums_mem(NUM_BLOCK, NoInit{});
    // kernel
    constexpr unsigned NUM_WARP = CEIL(NUM_THREAD, WARP_SIZE);
    softmax_online_1<float, NUM_WARP><<<NUM_BLOCK,NUM_THREAD>>>(N, x_mem.dev_p, maxs_mem.dev_p, exp_sums_mem.dev_p);
    softmax_online_2<float, NUM_WARP><<<NUM_BLOCK, NUM_THREAD>>>(N, x_mem.dev_p, logits_mem.dev_p,
                                                                 NUM_BLOCK, maxs_mem.dev_p, exp_sums_mem.dev_p);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // 写回
    logits_mem.deviceToHost();
}

void call_softmax_naive(const unsigned N, const DeviceMemory<float, true> &x_mem,
                        const DeviceMemory<float, true> &logits_mem) {
    // kernel
    constexpr unsigned NUM_THREAD = 256, PARALLEL_BLOCK_PER_SM = NUM_THREAD_PER_SM / NUM_THREAD,
            NUM_BLOCK = PARALLEL_BLOCK_PER_SM * NUM_SM;
    constexpr unsigned NUM_WARP = CEIL(NUM_THREAD, WARP_SIZE);
    const DeviceMemory<float, true> max_mem(1);
    const DeviceMemory<float, true> exp_sum_mem(1);
    float *max_p = max_mem.dev_p; // lambda使用
    float *exp_sum_p = exp_sum_mem.dev_p;
    // reduce max
    reduce<float, IdentityFunctor<float>, MaxFunctor<float>, AtomicMaxFunctor<float>, NUM_WARP><<<NUM_BLOCK,NUM_THREAD>>
            >(N, x_mem.dev_p, max_p);
    max_mem.deviceToHost();
    std::cout << max_mem.p[0] << std::endl;
    // reduce sum
    auto exp_func = [=] __device__ (const float x) { return expf(x - *max_p); };
    reduce<float, decltype(exp_func), AddFunctor<float>, AtomicAddFunctor<float>, NUM_WARP><<<NUM_BLOCK,NUM_THREAD>>>(
        N, x_mem.dev_p, exp_sum_p, exp_func);
    exp_sum_mem.deviceToHost();
    std::cout << exp_sum_mem.p[0] << std::endl;
    // element wise
    auto softmax_func = [=] __device__ (const float x) { return expf(x - *max_p) / *exp_sum_p; };
    element_wise<float, decltype(softmax_func)><<<NUM_BLOCK,NUM_THREAD>>>(
        N, x_mem.dev_p, logits_mem.dev_p, softmax_func);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    // 写回
    logits_mem.deviceToHost();
}
