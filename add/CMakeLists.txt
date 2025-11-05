add_executable(add1 add1.cu ../include/kernel.cu ../include/check.cu)
add_executable(add2 add2.cu ../include/kernel.cu ../include/check.cu)
add_executable(add3 add3.cu ../include/kernel.cu ../include/check.cu)
add_executable(add4 add4.cu ../include/kernel.cu ../include/check.cu)

set_target_properties(add1 add2 add3 add4
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
