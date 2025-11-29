add_executable(sgemm sgemm.cu call.cu kernel.cu ../util/util.cu)
set_target_properties(sgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sgemm PRIVATE cublas)
