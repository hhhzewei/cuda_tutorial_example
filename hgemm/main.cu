#include <random>
#include <map>
#include <cutlass/half.h>
#include "util/util.h"

#include "hgemm/call.h"
#include "util/device_memory.h"

int main(int argc, char *argv[]) {
    using T = cute::half_t;
    constexpr unsigned M = 1 << 11, N = 1 << 11, K = 1 << 11;

    // 1. 硬件随机数发生器，用来做种子
    std::random_device rd;
    // 2. 使用种子初始化伪随机数引擎 (Mersenne Twister)
    std::mt19937 gen(rd());
    // 3. 定义分布范围
    std::uniform_real_distribution distr(-1.0f, 1.0f);
    auto initializer_a = [&](unsigned i) { return T(distr(gen)); };
    auto initializer_b = [&](unsigned i) { return T(distr(gen)); };
    // auto initializer_a = [&](unsigned i) { return T(2.0f); };
    // auto initializer_b = [&](unsigned i) { return T(3.0f); };
    auto initializer_c = [&](unsigned i) { return T(0.0f); };
    const DeviceMemory<T, true> a_mem(M * K, initializer_a);
    const DeviceMemory<T, true> b_mem(K * N, initializer_b);
    const DeviceMemory<T, true> c_mem(M * N, initializer_c);
    std::cout << "Input generate finished" << std::endl;

    call_hgemm_sm80_cute<T>(M, K, N, a_mem.dev_p, b_mem.dev_p, c_mem.dev_p, c_mem.p);
    std::cout << "hgemm_sm80_cute error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, c_mem.p) << std::endl;

    call_hgemm_cublas<T>(M, K, N, a_mem.dev_p, b_mem.dev_p, c_mem.dev_p, c_mem.p);
    std::cout << "hgemm_cublas error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, c_mem.p) << std::endl;

    call_hgemm_cutlass<cutlass::half_t>(M, K, N, a_mem.dev_p, b_mem.dev_p, c_mem.dev_p, c_mem.p);
    std::cout << "hgemm_cutlass error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, c_mem.p) << std::endl;

    call_hgemm_cute_example<T>(M, K, N, a_mem.dev_p, b_mem.dev_p, c_mem.dev_p, c_mem.p);
    std::cout << "call_hgemm_cute_example error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, c_mem.p) << std::endl;


    // std::map<float, int> count_map;
    // for (int i = 0; i < M * N; ++i) {
    //     ++count_map[static_cast<float>(c_mem.p[i])];
    // }
    // for (auto [val,num]: count_map) {
    //     std::cout << val << ": " << num << std::endl;
    // }
}
