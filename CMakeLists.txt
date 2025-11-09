cmake_minimum_required(VERSION 3.5)
project(cuda_tutorial_example CUDA)

#set(CMAKE_CUDA_ARCHITECTURES 75)# T4
set(CMAKE_CUDA_ARCHITECTURES 80)# A100

set(CMAKE_CUDA_STANDARD 11)
include_directories(${PROJECT_SOURCE_DIR}/include)
#add_subdirectory(add)
#
#add_executable(add main/add.cu include/call.cu include/kernel.cu include/check.cu)
#set_target_properties(add PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(add PRIVATE cublas)
#
#
#add_executable(dot main/dot.cu include/call.cu include/kernel.cu include/check.cu)
#set_target_properties(dot PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(dot PRIVATE cublas)
#
#
#add_executable(transpose main/transpose.cu include/call.cu include/kernel.cu include/check.cu)
#set_target_properties(transpose PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#target_link_libraries(transpose PRIVATE cublas)

add_executable(sgemm main/sgemm.cu include/call.cu include/kernel.cu include/check.cu)
set_target_properties(sgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(sgemm PRIVATE cublas)


