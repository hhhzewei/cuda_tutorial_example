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

int getSmCount(int device_id) {
    cudaGetDevice(&device_id);
    int sm_count = 0;
    // 查询 SM 数量
    check_error(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id));
    return sm_count;
}
