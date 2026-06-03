add_executable(hgemm main.cu ../util/util.cu)
set_target_properties(hgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(hgemm PRIVATE cublas)
target_link_libraries(hgemm PRIVATE nvidia::cutlass::cutlass)
target_compile_options(hgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -lineinfo >)

if (MSVC)
    add_executable(test test.cu)
    target_link_libraries(test PRIVATE nvidia::cutlass::cutlass)
endif ()

