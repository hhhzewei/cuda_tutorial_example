//
// Created by hzw on 2026/1/1.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_FUNCTOR_H
#define CUDA_TUTORIAL_EXAMPLE_FUNCTOR_H
#include "util.h"

template<typename T>
struct IdentityFunctor {
    __device__ __forceinline__ T operator()(T a) {
        return a;
    }
};

template<typename T>
struct AddFunctor {
    __device__ __forceinline__ constexpr static T init() {
        return 0;
    }

    __device__ __forceinline__ T operator()(T a, T b) {
        return a + b;
    }
};

template<typename T>
struct MultipleFunctor {
    __device__ __forceinline__ constexpr static T init() {
        return 1;
    }

    __device__ __forceinline__ T operator()(T a, T b) {
        return a * b;
    }
};

template<typename T>
struct MaxFunctor {
    __device__ __forceinline__ constexpr static T init() {
        return my_math::min_value<T>();
    }

    __device__ __forceinline__ T operator()(T a, T b) {
        return my_math::fmax(a, b);
    }
};


__device__ __forceinline__ float atomicMaxf(float *p, float value) {
    int *p_int = (int *) p;
    int old_int = *p_int, assume_int;
    do {
        assume_int = old_int;
        const float res = fmaxf(__int_as_float(assume_int), value);
        old_int = atomicCAS(p_int, assume_int, __float_as_int(res));
    } while (assume_int != old_int); // 比较int避开float比较

    return __int_as_float(old_int);
}

template<typename T>
struct AtomicMaxFunctor {
    __device__ __forceinline__ T operator()(T *p, T value) {
        return atomicMaxf(p, value);
    }
};


template<typename T>
struct AtomicAddFunctor {
    __device__ __forceinline__ void operator()(T *p, T value) {
        atomicAdd(p, value);
    }
};
#endif //CUDA_TUTORIAL_EXAMPLE_FUNCTOR_H
