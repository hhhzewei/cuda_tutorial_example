#include <iostream>
#include "kernel.h"
#include "check.h"

// 单一线程
int main(void)
{
    int N = 1<<20;
    float *x, *y, *ret;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));
    cudaMallocManaged(&ret, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    add<<<1, 1>>>(N, x, y, ret);
    CHECK_ERROR

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    std::cout << "Max error: " << add_error(N, x, y, ret)<< std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(ret);
    return 0;
}