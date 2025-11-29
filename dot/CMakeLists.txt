add_executable(dot dot.cu call.cu kernel.cu ../util/util.cu)
set_target_properties(dot PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(dot PRIVATE cublas)
