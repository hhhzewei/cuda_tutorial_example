#pragma once
#include <cublas_v2.h>

#include <cutlass/detail/layout.hpp>
#include <cutlass/gemm/device/gemm.h>

#include "hgemm/hgemm_sm80_cute.h"
#include "hgemm/hgemm_sm80_cute_example.h"
#include "util/util.h"

template<typename T>
void call_hgemm_cublas(const int M, const int K, const int N,
                       const T *dev_a, const T *dev_b, T *dev_c, T *ret) {
    // 1. 创建 cuBLAS 句柄
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS 初始化失败！" << std::endl;
        return;
    }
    // 2. 混合精度所需的缩放因子 (FP32 累加)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // 默认TN布局
    // 原始 A 是转置(T)，转置的转置变成不转置 -> CUBLAS_OP_N
    // 原始 B 是不转置(N)，现在变成转置 -> CUBLAS_OP_T
    // 交换位置后：左矩阵是 B，右矩阵是 A
    // 维度 M 和 N 在参数列表中对调位置
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_T, // 针对 B：原始是不转置，公式推导后需要转置
        CUBLAS_OP_N, // 针对 A：原始是转置，转置的转置变成不转置
        N, M, K, // 【修改】M 和 N 对调
        &alpha,
        dev_b, CUDA_R_16F, K, // 【修改】左矩阵换成 B，主维度 ldb = K (因为转置了)
        dev_a, CUDA_R_16F, K, // 【修改】右矩阵换成 A，主维度 lda = K (不转置，行数是K)
        &beta,
        dev_c, CUDA_R_16F, N, // 【修改】输出矩阵 C，主维度 ldc = N (此时对 cuBLAS 来说行数是 N)
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cublasGemmEx 执行失败，错误码: " << status << std::endl;
    }
    cudaMemcpy(ret, dev_c, M * N * sizeof(T), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
}

template<typename T=cute::half_t, typename =std::enable_if_t<cute::is_any_of_v<T, cute::half_t, cute::bfloat16_t> > >
void call_hgemm_cutlass(const int M, const int K, const int N,
                        const T *d_a, const T *d_b, T *d_c, T *ret) {
    // ================== CUTLASS 类型定义 ==================
    using ElementInputA = T;
    using ElementInputB = T;
    using ElementOutput = T;
    using ElementAccumulator = float; // 保持 FP32 混合精度累加

    using LayoutA = cutlass::layout::RowMajor; // T -> 物理行主序
    using LayoutB = cutlass::layout::ColumnMajor; // N -> 物理列主序
    using LayoutC = cutlass::layout::RowMajor; // 输出行主序

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 32>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;
    using ShapeMMAMma = cutlass::gemm::GemmShape<16, 8, 16>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementAccumulator
    >;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ShapeMMAThreadBlock,
        ShapeMMAWarp,
        ShapeMMAMma,
        EpilogueOp
    >;

    cutlass::gemm::GemmCoord problem_size(M, N, K);

    // 计算主维步长 (Leading Dimension)
    int lda = K;
    int ldb = K;
    int ldc = N;

    typename Gemm::TensorRefA ref_A(d_a, LayoutA::packed({M, K}).stride(lda));

    typename Gemm::TensorRefB ref_B(d_b, LayoutB::packed({K, N}).stride(ldb));

    // C 和 D 均指向同一块本地显存 d_C
    typename Gemm::TensorRefC ref_C(const_cast<T *>(d_c), LayoutC::packed({M, N}).stride(ldc));
    typename Gemm::TensorRefD ref_D(d_c, LayoutC::packed({M, N}).stride(ldc));

    // 构造参数
    typename Gemm::Arguments arguments(
        problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D
    );

    // 初始化并运行
    Gemm gemm_op;
    cutlass::Status status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        return;
    }

    status = gemm_op();
    if (status != cutlass::Status::kSuccess) return;
    cudaMemcpy(ret, d_c, M * N * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void call_hgemm_sm80_cute(const int M, const int K, const int N,
                          const T *dev_a, const T *dev_b, T *dev_c, T *ret) {
    constexpr int kBlockM = 128, kBlockN = 128, kBlockK = 64;
    constexpr int kMmaWarpLayoutM = 2, kMmaWarpLayoutN = 2;
    constexpr int kCopyThreadLayoutM = 16, kCopyThreadLayoutN = 8;
    constexpr int kNumThread = kCopyThreadLayoutM * kCopyThreadLayoutN;
    constexpr int kNumPipe = 2;
    constexpr int kSMemSize = sizeof(T) * std::max(kBlockK * (kBlockM + kBlockN) * kNumPipe,
                                                   kBlockM * kBlockN);
    static_assert(kSMemSize <= 102400); //共享内存上界
    dim3 grid_dim{static_cast<unsigned int>(M / kBlockM), static_cast<unsigned int>(N / kBlockN)};
    auto kernel_fptr = hgemm_sm80_cute<T, kBlockM, kBlockN, kBlockK, kNumPipe,
        kCopyThreadLayoutM, kCopyThreadLayoutN,
        kMmaWarpLayoutM, kMmaWarpLayoutN>;
    // Set L1 to be SMEM only
    if (kSMemSize > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel_fptr,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSMemSize);
    }

    kernel_fptr<<<grid_dim,kNumThread,kSMemSize>>>(
        dev_a, dev_b, dev_c, M, N, K);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_c, M * N * sizeof(T), cudaMemcpyDeviceToHost);
}


template<typename T>
void call_hgemm_cute_example(const int M, const int K, const int N,
                             const T *dev_a, const T *dev_b, T *dev_c, T *ret) {
    gemm_tn_cute_example(M, N, K, 1.0,
                         dev_a, K,
                         dev_b, K,
                         0.0f,
                         dev_c, N);
    check_error(cudaGetLastError());
    check_error(cudaDeviceSynchronize());
    cudaMemcpy(ret, dev_c, M * N * sizeof(T), cudaMemcpyDeviceToHost);
}
