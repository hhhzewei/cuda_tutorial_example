add_executable(add add.cu call.cu kernel.cu ../util/util.cu)
set_target_properties(add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(add PRIVATE cublas)
