add_executable(softmax softmax.cu call.cu ../util/util.cu)
set_target_properties(softmax PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(softmax PRIVATE "${TORCH_LIBRARIES}")
target_compile_options(softmax PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
