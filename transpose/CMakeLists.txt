add_executable(transpose transpose.cu call.cu kernel.cu ../util/util.cu)
set_target_properties(transpose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(transpose PRIVATE cublas)