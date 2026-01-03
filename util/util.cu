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