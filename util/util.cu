//
// Created by hzw on 2025/10/27.
//

#include "util.h"
#include <cstdio>

void check_error(const cudaError_t err) {
    if (err != cudaSuccess) {
        printf("Error:%s\n", cudaGetErrorString(err));
    }
}

void batch_free(const std::initializer_list<void *> ptr_list) {
    for (const auto p: ptr_list) {
        cudaFree(p);
    }
}