//
// Created by hzw on 2025/10/27.
//

#ifndef CUDA_TUTORIAL_EXAMPLE_CHECK_CUH
#define CUDA_TUTORIAL_EXAMPLE_CHECK_CUH

#define CHECK_ERROR cudaError_t err=cudaGetLastError();    \
if(err != cudaSuccess){    \
printf("Error:%s\n",cudaGetErrorString(err));    \
}    \
else{    \
printf("CudaSuccess\n"); \
}

void check_error(cudaError_t err);

float add_error(unsigned N, float* a, float*b, float*c);

float dot_error(unsigned N, float* a, float *b, float *ret);

float transpose_error(unsigned M, unsigned N, float *input, float *output);

float sgemm_error(unsigned M, unsigned K, unsigned N, float *a, float *b, float *ret);
#endif //CUDA_TUTORIAL_EXAMPLE_CHECK_CUH