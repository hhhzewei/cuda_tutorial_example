
#include "cute/algorithm/cooperative_copy.hpp"

int main(int argc, char *argv[]) {
    using namespace cute;
    auto tiled_mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
                                    Layout<Shape<_2, _2>>{});
    auto thr_mma = tiled_mma.(1);
    Tensor tCgA=thr_mma.partion_A(sA)
}

