//
// Created by hzw on 2026/1/1.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_FUNCTOR_H
#define CUDA_TUTORIAL_EXAMPLE_FUNCTOR_H

template<typename T>
struct  AddFunctor {
    __device__ __forceinline__ T operator()(T a, T b) {
        return a + b;
    }
};

template<typename T>
struct MultipleFunctor {
    __device__ __forceinline__ T operator()(T a, T b) {
        return a * b;
    }
};

template<typename T>
struct AtomicAddFunctor {
    __device__ __forceinline__ void operator()(T *p, T value) {
        atomicAdd(p, value);
    }
};
#endif //CUDA_TUTORIAL_EXAMPLE_FUNCTOR_H
