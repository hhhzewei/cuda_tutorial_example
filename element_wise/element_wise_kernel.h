//
// Created by hzw on 2026/1/1.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_ELEMENT_WISE_KERNEL_H
#define CUDA_TUTORIAL_EXAMPLE_ELEMENT_WISE_KERNEL_H

template<typename T, template<typename> class CalFunc>
__global__ void element_wise(const unsigned N, T *a, T *b, T *c) {
    CalFunc<T> cal_func{};
    const unsigned threadIdxGlobal = threadIdx.x + blockIdx.x * blockDim.x, NUM_THREAD = gridDim.x * blockDim.x;
    for (unsigned i = threadIdxGlobal; i < N; i += NUM_THREAD) {
        c[i] = cal_func(a[i], b[i]);
    }
}

#endif //CUDA_TUTORIAL_EXAMPLE_ELEMENT_WISE_KERNEL_H
