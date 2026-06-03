#include <iostream>
#include "call.h"
#include "sgemm/gemm_sm80_cuda.h"
#include "util/device_memory.h"
#include "util/util.h"


int main() {
    constexpr unsigned M = 1 << 11, N = 1 << 11, K = 1 << 12;
    auto initializer_a = [](unsigned i) { return 0.1f; };
    auto initializer_b = [](unsigned i) { return 0.2f; };
    const DeviceMemory<float, true> a_mem(M * K, initializer_a);
    const DeviceMemory<float, true> b_mem(K * N, initializer_b);
    const DeviceMemory<float, true> ret_mem(M * N);
    // // call cublas sgemm
    // call_sgemm_cublas(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call sgemm cublas kernel: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call cutlass sgemm colab装cutlass太麻烦了
    // // call sgemm naive kernel
    // call_sgemm_naive(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "sgemm naive error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm block tile kernel
    // call_sgemm_block_tile(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "sgemm block tile error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm thread tile kernel
    // call_sgemm_thread_tile_v0(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "sgemm thread tile v0 error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm thread tile v1 kernel太慢永久封印
    // // call sgemm thread tile v2 kernel
    // call_sgemm_thread_tile_v2(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call sgemm thread tile v2 padding error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm thread tile v3 kernel
    // call_sgemm_thread_tile_v3(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call sgemm thread tile v3 error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm thread tile v4 kernel
    // call_sgemm_thread_tile_v4(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call sgemm thread tile v4 error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm thread tile v5 kernel
    // call_sgemm_thread_tile_v5(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call sgemm thread tile v5 error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm thread tile v6 kernel
    // call_sgemm_thread_tile_v6(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call sgemm thread tile v6 error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm thread tile v7 kernel
    // call_sgemm_thread_tile_v7(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call sgemm thread tile v7 error: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm tensor core v0 kernel 性能不行，一个block一个warp也太蠢了
    // // call sgemm tensor core v1 kernel
    // call_sgemm_tensor_core_v1(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call sgemm tensor core v1 kernel: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // // call sgemm tensor core v2 kernel
    // call_sgemm_tensor_core_v2(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call sgemm tensor core v2 kernel: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call gemm kernel
    // call_gemm(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    // std::cout << "call gemm kernel: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
    // call gemm sm80 kernel
    call_gemm_sm80_cuda(M, K, N, a_mem.dev_p, b_mem.dev_p, ret_mem.dev_p, ret_mem.p);
    std::cout << "call gemm sm80 kernel: " << gemm_error(M, K, N, a_mem.p, b_mem.p, ret_mem.p) << std::endl;
}
