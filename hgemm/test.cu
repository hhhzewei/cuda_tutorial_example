#include <random>
#include <cute/tensor_impl.hpp>
#include <cute/atom/copy_atom.hpp>
using T = cute::half_t;


__global__ void hgemm_sm80_cute(const T *a, const T *b, half *c) {
    using namespace cute;
    constexpr auto M = _1024{};
    constexpr auto N = _1024{};
    constexpr auto K = _1024{};
    // 块大小与线程布局常量定义
    constexpr int kBlockM = 128;
    constexpr int kBlockN = 128;
    constexpr int kBlockK = 64;

    constexpr int kCopyThreadLayoutM = 32;
    constexpr int kCopyThreadLayoutN = 8;

    constexpr int kMmaWarpLayoutM = 2;
    constexpr int kMmaWarpLayoutN = 4;
    CUTE_STATIC_ASSERT(kMmaWarpLayoutM * kMmaWarpLayoutN * 32 == kCopyThreadLayoutM * kCopyThreadLayoutN);

    
    using _kBlockM = Int<kBlockM>;
    using _kBlockN = Int<kBlockN>;
    using _kBlockK = Int<kBlockK>;
    const int thr_block_x = static_cast<int>(blockIdx.x), thr_block_y = static_cast<int>(blockIdx.y);
    const int thread_idx = static_cast<int>(threadIdx.x);
    Tensor mA = make_tensor(make_gmem_ptr(a), make_layout(make_shape(M, K), LayoutRight{}));
    Tensor mB = make_tensor(make_gmem_ptr(b), make_layout(make_shape(N, K), LayoutRight{}));
    Tensor mC = make_tensor(make_gmem_ptr(c), make_layout(make_shape(M, N), LayoutRight{}));
    Tensor gA = local_tile(mA, make_tile(_kBlockM{}, _kBlockK{}), make_coord(thr_block_x, _)); // BLK_M,BLK_K,NUM_BLK_K
    Tensor gB = local_tile(mB, make_tile(_kBlockN{}, _kBlockK{}), make_coord(thr_block_y, _)); // BLK_N,BLK_K,NUM_BLK_K
    Tensor gC = local_tile(mC, make_tile(_kBlockM{}, _kBlockN{}), make_coord(thr_block_x, thr_block_y)); // BLK_M,BLK_N

    // __shared__ T s_a[2 * kBlockM * kBlockK];
    // __shared__ T s_b[2 * kBlockN * kBlockK];
    extern __shared__ T s_mem[]; // 动态共享内存，上界更高
    T *s_a = s_mem;
    T *s_b = s_mem + 2 * kBlockM * kBlockK;
    T *s_c = s_mem;
    // Tensor sA = make_tensor(make_smem_ptr(s_a),
    //                         flatten(logical_product(make_layout(Shape<_kBlockM, _kBlockK>{}, LayoutRight{}), _2{})));
    // Tensor sB = make_tensor(make_smem_ptr(s_b),
    //                         flatten(logical_product(make_layout(Shape<_kBlockN, _kBlockK>{}, LayoutRight{}), _2{})));
    // swizzle
    auto swizzle_atom = composition(Swizzle<3, 3, 3>{},
                                    Layout<Shape<_8, Shape<_8, _8> >, Stride<_64, Stride<_1, _8> > >{});
    auto sA_layout = tile_to_shape(swizzle_atom, Shape<_kBlockM, _kBlockK, _2>{});
    auto sB_layout = tile_to_shape(swizzle_atom, Shape<_kBlockN, _kBlockK, _2>{});
    auto sC_layout = make_layout(Shape<_kBlockM, _kBlockN>{}, LayoutRight{});
    Tensor sA = make_tensor(make_smem_ptr(s_a), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(s_b), sB_layout);
    Tensor sC = make_tensor(make_smem_ptr(s_c), sC_layout);

    // block tile必须整数倍于copy tile
    CUTE_STATIC_ASSERT(kBlockM % (kCopyThreadLayoutM * 1) == 0);
    CUTE_STATIC_ASSERT(kBlockN % (kCopyThreadLayoutM * 1) == 0);
    CUTE_STATIC_ASSERT(kBlockK % (kCopyThreadLayoutN * 8) == 0);
    TiledCopy tiled_copy_in = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, T>{},
                                              make_layout(Shape<Int<kCopyThreadLayoutM>, Int<kCopyThreadLayoutN> >{},
                                                          LayoutRight{}),
                                              make_layout(Shape<_1, _8>{}, LayoutRight{}));
    auto thr_copy_in = tiled_copy_in.get_slice(thread_idx);
    Tensor tAgA = thr_copy_in.partition_S(gA); // CPY,CPY_M,CPY_K,NUM_BLK_K
    Tensor tAsA = thr_copy_in.partition_D(sA); // CPY,CPY_M,CPY_K,2
    Tensor tBgB = thr_copy_in.partition_S(gB);
    Tensor tBsB = thr_copy_in.partition_D(sB);
    // first load
    copy(tiled_copy_in, tAgA(_, _, _, _0{}), tAsA(_, _, _, _0{}));
    copy(tiled_copy_in, tBgB(_, _, _, _0{}), tBsB(_, _, _, _0{}));
    cp_async_fence();

    // SM75_U32x4_LDSM_N需要一个warp搬运16*16 half
    // mma在B上一个warp是16*8，所以permutation的N需要多一倍，否则s2rCopy构建出错
    using MMA = std::conditional_t<is_any_of_v<T, half_t, half>,
        SM80_16x8x16_F32F16F16F32_TN,
        SM80_16x8x16_F32BF16BF16F32_TN>;
    auto mma_permutation = Shape<Int<kMmaWarpLayoutM * 16>, Int<kMmaWarpLayoutN * 8 * 2>, _16>{};
    CUTE_STATIC_ASSERT(kBlockM % get<0>(mma_permutation) == 0);
    CUTE_STATIC_ASSERT(kBlockN % get<1>(mma_permutation) == 0);
    CUTE_STATIC_ASSERT(kBlockK % get<2>(mma_permutation) == 0);
    TiledMMA tiled_mma = make_tiled_mma(MMA_Atom<MMA>{},
                                        Layout<Shape<Int<kMmaWarpLayoutM>, Int<kMmaWarpLayoutN> > >{},
                                        Shape<Int<kMmaWarpLayoutM * 16>,
                                            Int<kMmaWarpLayoutN * 8 * 2>, _16>{});
    ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);
    Tensor tCsC = thr_mma.partition_C(sC); // MMA,MMA_M,MMA_N
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, _0{})); // MMA,MMA_M,MMA_K
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, _0{})); // MMA,MMA_N,MMA_K
    Tensor tCrC = thr_mma.make_fragment_C(tCsC); // MMA,MMA_M,MMA_N

    TiledCopy tiled_s2r_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, tiled_mma);
    ThrCopy thr_s2r_copy_A = tiled_s2r_copy_A.get_slice(thread_idx);
    Tensor tXsA = thr_s2r_copy_A.partition_S(sA); // MMA,MMA_M,MMA_K,2
    Tensor tXrA = thr_s2r_copy_A.retile_D(tCrA); //MMA,MMA_M,MMA_K


    TiledCopy tiled_s2r_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, tiled_mma);
    ThrCopy thr_s2r_copy_B = tiled_s2r_copy_B.get_slice(thread_idx);
    Tensor tXsB = thr_s2r_copy_B.partition_S(sB); // MMA,MMA_N,MMA_K,2
    Tensor tXrB = thr_s2r_copy_B.retile_D(tCrB); // MMA,MMA_N,MMA_K

    int pipe_idx = 1;
    int tile_idx = 1;
    clear(tCrC);
    for (; tile_idx < size<2>(gA); ++tile_idx) {
        // load
        copy(tiled_copy_in, tAgA(_, _, _, tile_idx), tAsA(_, _, _, pipe_idx));
        copy(tiled_copy_in, tBgB(_, _, _, tile_idx), tBsB(_, _, _, pipe_idx));
        cp_async_fence(); // 提交cp sync计数不能遗漏
        // compute
        cp_async_wait<1>();
        __syncthreads();
        pipe_idx = 1 - pipe_idx;

        // for (int i = 0; i < size(tAsA(_, _, _, pipe_idx)); ++i) {
        //     if (tAsA(_, _, _, pipe_idx)(i) != 1) {
        //         printf("A: %i %f\n", &tAsA(_, _, _, pipe_idx)(i) - s_a, float(tAsA(_, _, _, pipe_idx)(i)));
        //         assert(&tAsA(_, _, _, pipe_idx)(i)>=s_a && &tAsA(_, _, _, pipe_idx)(i) < s_a+kBlockM*kBlockK*2);
        //         assert(tAsA(_, _, _, pipe_idx)(i)==1);
        //     }
        // }
        // for (int i = 0; i < size(tBsB(_, _, _, pipe_idx)); ++i) {
        //     if (tBsB(_, _, _, pipe_idx)(i) != 2) {
        //         printf("B: %i %f\n", &tBsB(_, _, _, pipe_idx)(i) - s_b, float(tBsB(_, _, _, pipe_idx)(i)));
        //         assert(&tBsB(_, _, _, pipe_idx)(i)>=s_b && &tBsB(_, _, _, pipe_idx)(i) < s_b+kBlockN*kBlockK*2);
        //         assert(tBsB(_, _, _, pipe_idx)(i)==2);
        //     }
        // }
#pragma unroll
        for (int mma_k_idx = 0; mma_k_idx < size<2>(tCrA); ++mma_k_idx) {
            copy(tiled_s2r_copy_A, tXsA(_, _, mma_k_idx, pipe_idx), tXrA(_, _, mma_k_idx));
            copy(tiled_s2r_copy_B, tXsB(_, _, mma_k_idx, pipe_idx), tXrB(_, _, mma_k_idx));
            gemm(tiled_mma, tCrA(_, _, mma_k_idx), tCrB(_, _, mma_k_idx), tCrC);
        }
        __syncthreads();
    }
    // last copy
    cp_async_wait<0>();
    __syncthreads();
    pipe_idx = 1 - pipe_idx;
#pragma unroll
    for (int mma_k_idx = 0; mma_k_idx < size<2>(tCrA); ++mma_k_idx) {
        copy(tiled_s2r_copy_A, tXsA(_, _, mma_k_idx, pipe_idx), tXrA(_, _, mma_k_idx));
        copy(tiled_s2r_copy_B, tXsB(_, _, mma_k_idx, pipe_idx), tXrB(_, _, mma_k_idx));
        gemm(tiled_mma, tCrA(_, _, mma_k_idx), tCrB(_, _, mma_k_idx), tCrC);
    }

    // write
#pragma unroll
    for (int i = 0; i < size(tCrC); i += 2) {
        const float2 float2_vec = make_float2(tCrC(i), tCrC(i + 1));
        if constexpr (is_any_of_v<T, half, half_t>) {
            // *reinterpret_cast<half2 *>(&tCgC(i)) = __float22half2_rn(float2_vec);
            *reinterpret_cast<half2 *>(&tCsC(i)) = __float22half2_rn(float2_vec);
        } else {
            // *reinterpret_cast<nv_bfloat162 *>(&tCgC(i)) = __float22bfloat162_rn(float2_vec);
            *reinterpret_cast<nv_bfloat162 *>(&tCsC(i)) = __float22bfloat162_rn(float2_vec);
        }
    }
    __syncthreads();

    auto tiled_copy_out = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, T>{},
                                          make_layout(Shape<Int<kCopyThreadLayoutM>, Int<kCopyThreadLayoutN> >{},
                                                      LayoutRight{}),
                                          make_layout(Shape<_1, _8>{}, LayoutRight{}));
    auto thr_copy_out = tiled_copy_out.get_slice(thread_idx);
    Tensor tYsC = thr_copy_out.partition_S(sC);
    Tensor tYgC = thr_copy_out.partition_D(gC);
    copy(tYsC, tYgC);
}


int main(int argc, char *argv[]) {
    using T = half;
    constexpr unsigned M = 1 << 10, N = 1 << 10, K = 1 << 10;
    // 1. 硬件随机数发生器，用来做种子
    std::random_device rd;
    // 2. 使用种子初始化伪随机数引擎 (Mersenne Twister)
    std::mt19937 gen(rd());
    // 3. 定义分布范围
    std::uniform_real_distribution distr(-1.0f, 1.0f);

    std::unique_ptr<T[], decltype(&free)> a_mem(static_cast<T *>(malloc(M * K * sizeof(T))), &free);
    std::unique_ptr<T[], decltype(&free)> b_mem(static_cast<T *>(malloc(N * K * sizeof(T))), &free);
    std::unique_ptr<T[], decltype(&free)> c_mem(static_cast<T *>(malloc(N * M * sizeof(T))), &free);
    for (int i = 0; i < M * K; ++i) a_mem[i] = T(distr(gen));
    for (int i = 0; i < N * K; ++i) b_mem[i] = T(distr(gen));
    for (int i = 0; i < M * N; ++i) c_mem[i] = 0;
    // for (int i = 0; i < M; ++i) {
    //     for (int j = 0; j < N; ++j) {
    //         for (int k = 0; k < K; ++k) {
    //             c_mem[i * N + j] += a_mem[i * K + k] * b_mem[k * N + j];
    //         }
    //     }
    // }
    // std::cout << "hgemm_cublas error: " << gemm_error(M, K, N, a_mem.get(), b_mem.get(), c_mem.get()) << std::endl;
    half x = 10;
    std::cout << float(x) << std::endl;

    constexpr auto test=std::isinf(fabsf(12288-12288));
}
