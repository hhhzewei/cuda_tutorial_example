//
// Created by hzw on 2026/1/1.
//
#include <torch/torch.h>
#include "call.h"

void call_softmax_torch(unsigned N, float *dev_x, float *logits) {
    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
    torch::Tensor input = torch::from_blob(dev_x, {1, N}, options);
    // 执行运算
    torch::Tensor result = torch::softmax(input, 1);
    // 拷贝结果回主机
    cudaMemcpy(logits, result.data_ptr<float>(), N * sizeof(float), cudaMemcpyDeviceToHost);
}
