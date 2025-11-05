#include <iostream>
#include "kernel.h"
#include "check.h"

// 预处理统一内存
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

    // Prefetch the x and y arrays to the GPU
    cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
    cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    std::cout<<"numBlocks: "<<numBlocks<<std::endl;
    add<<<numBlocks, blockSize>>>(N, x, y, ret);
    CHECK_ERROR
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    std::cout << "Max error: " << add_error(N,x,y,ret)<< std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(ret);
    return 0;
}