#pragma once


#include "cute/tensor.hpp"

template<typename T = cute::half_t, int kBlockM = 128, int kBlockN = 128, int kBlockK = 64, int kNumPipe = 2,
    int kCopyThreadLayoutM = 16, int kCopyThreadLayoutN = 16,
    int kMmaWarpLayoutM = 2, int kMmaWarpLayoutN = 4,
    typename = std::enable_if_t<cute::is_any_of_v<T, cute::half_t, half, cute::bfloat16_t, nv_bfloat16> > >
__global__ void hgemm_sm80_cute(const T *a, const T *b, T *c, const int M, const int N, const int K) {
    using namespace cute;
    CUTE_STATIC_ASSERT(kMmaWarpLayoutM * kMmaWarpLayoutN * 32 == kCopyThreadLayoutM * kCopyThreadLayoutN);
    using _kBlockM = Int<kBlockM>;
    using _kBlockN = Int<kBlockN>;
    using _kBlockK = Int<kBlockK>;
    using _kNumPipe = Int<kNumPipe>;

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
    T *s_b = s_mem + kNumPipe * kBlockM * kBlockK;
    T *s_c = s_mem;
    // swizzle
    // 因为ldmatrix会搬运4个8x8共享内存tile，造成四次8-way bank conflict, ncu上显示合并成32-way bank conflict

    auto swizzle_atom_AB = composition(Swizzle<3, 3, 3>{},
                                       Layout<Shape<_8, Shape<_8, _8> >, Stride<_64, Stride<_1, _8> > >{});
    auto swizzle_atom_C = composition(Swizzle<3, 1, 5>{},

                                      Layout<Shape<_8, Shape<_2, _32> >, Stride<_64, Stride<_1, _2> > >{});
    // auto sA_layout = flatten(logical_product(make_layout(Shape<_kBlockM, _kBlockK>{}, LayoutRight{}), _kNumPipe{}));
    // auto sB_layout = flatten(logical_product(make_layout(Shape<_kBlockN, _kBlockK>{}, LayoutRight{}), _kNumPipe{}));
    // auto sC_layout = make_layout(Shape<_kBlockM, _kBlockN>{}, LayoutRight{});
    auto sA_layout = tile_to_shape(swizzle_atom_AB, Shape<_kBlockM, _kBlockK, _kNumPipe>{});
    auto sB_layout = tile_to_shape(swizzle_atom_AB, Shape<_kBlockN, _kBlockK, _kNumPipe>{});
    auto sC_layout = tile_to_shape(swizzle_atom_C, Shape<_kBlockM, _kBlockN>{});

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
#pragma unroll
    for (int pipe_idx = 0; pipe_idx < kNumPipe; ++pipe_idx) {
        copy(tiled_copy_in, tAgA(_, _, _, pipe_idx), tAsA(_, _, _, pipe_idx));
        copy(tiled_copy_in, tBgB(_, _, _, pipe_idx), tBsB(_, _, _, pipe_idx));
        cp_async_fence();
    }
    // SM75_U32x4_LDSM_N需要一个warp搬运16*16 half
    // mma在B上一个warp是8*16，所以permutation的N需要多一倍，否则s2rCopy构建出错
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
    Tensor tCrA = thr_mma.partition_fragment_A(sA(_, _, _0{})); // MMA,MMA_M,MMA_K
    Tensor tCrB = thr_mma.partition_fragment_B(sB(_, _, _0{})); // MMA,MMA_N,MMA_K

    // Tensor tCsC = thr_mma.partition_C(sC); // MMA,MMA_M,MMA_N
    // Tensor tCrC = thr_mma.make_fragment_C(tCsC); // MMA,MMA_M,MMA_N
    Tensor tCgC = thr_mma.partition_C(gC); // MMA,MMA_M,MMA_N
    Tensor tCrC = thr_mma.make_fragment_C(tCgC); // MMA,MMA_M,MMA_N

    TiledCopy tiled_s2r_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, tiled_mma);
    ThrCopy thr_s2r_copy_A = tiled_s2r_copy_A.get_slice(thread_idx);
    Tensor tXsA = thr_s2r_copy_A.partition_S(sA); // MMA,MMA_M,MMA_K,2
    Tensor tXrA = thr_s2r_copy_A.retile_D(tCrA); //MMA,MMA_M,MMA_K


    TiledCopy tiled_s2r_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, T>{}, tiled_mma);
    ThrCopy thr_s2r_copy_B = tiled_s2r_copy_B.get_slice(thread_idx);
    Tensor tXsB = thr_s2r_copy_B.partition_S(sB); // MMA,MMA_N,MMA_K,2
    Tensor tXrB = thr_s2r_copy_B.retile_D(tCrB); // MMA,MMA_N,MMA_K


    clear(tCrC);
    int pipe_idx = 0;
    for (int tile_idx = kNumPipe; tile_idx < size<2>(gA); ++tile_idx, pipe_idx = (pipe_idx + 1) % kNumPipe) {
        // compute
        cp_async_wait<kNumPipe - 1>();
        __syncthreads();
        copy(tiled_s2r_copy_A, tXsA(_, _, _, pipe_idx), tXrA);
        copy(tiled_s2r_copy_B, tXsB(_, _, _, pipe_idx), tXrB);
        gemm(tiled_mma, tCrA, tCrB, tCrC);

        __syncthreads();
        // load
        copy(tiled_copy_in, tAgA(_, _, _, tile_idx), tAsA(_, _, _, pipe_idx));
        copy(tiled_copy_in, tBgB(_, _, _, tile_idx), tBsB(_, _, _, pipe_idx));
        cp_async_fence(); // 提交cp sync计数不能遗漏
    }
    // rest compute
#pragma unroll
    for (int i = kNumPipe - 1; i >= 0; --i) {
        switch (i) {
            case 0: cp_async_wait<0>();
                break;
            case 1: cp_async_wait<1>();
                break;
            case 2: cp_async_wait<2>();
                break;
            case 3: cp_async_wait<3>();
                break;
            default: ;
        }
        __syncthreads();
        copy(tiled_s2r_copy_A, tXsA(_, _, _, pipe_idx), tXrA);
        copy(tiled_s2r_copy_B, tXsB(_, _, _, pipe_idx), tXrB);
        gemm(tiled_mma, tCrA, tCrB, tCrC);
        pipe_idx = (pipe_idx + 1) % kNumPipe;
    }

    // write
#pragma unroll
    for (int i = 0; i < size(tCrC); i += 2) {
        const float2 float2_vec = make_float2(tCrC(i), tCrC(i + 1));
        if constexpr (is_any_of_v<T, half, half_t>) {
            *reinterpret_cast<half2 *>(&tCgC(i)) = __float22half2_rn(float2_vec);
            // *reinterpret_cast<half2 *>(&tCsC(i)) = __float22half2_rn(float2_vec);
        } else {
            *reinterpret_cast<nv_bfloat162 *>(&tCgC(i)) = __float22bfloat162_rn(float2_vec);
            // *reinterpret_cast<nv_bfloat162 *>(&tCsC(i)) = __float22bfloat162_rn(float2_vec);
        }
    }

    // __syncthreads(); //写回全局内存读共享内存
    //
    // auto tiled_copy_out = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, T>{},
    //                                       make_layout(Shape<Int<kCopyThreadLayoutM>, Int<kCopyThreadLayoutN> >{},
    //                                                   LayoutRight{}),
    //                                       make_layout(Shape<_1, _8>{}, LayoutRight{}));
    // auto thr_copy_out = tiled_copy_out.get_slice(thread_idx);
    // Tensor tYsC = thr_copy_out.partition_S(sC);
    //
    // Tensor tYgC = thr_copy_out.partition_D(gC);
    // copy(tiled_copy_out, tYsC, tYgC);
}
