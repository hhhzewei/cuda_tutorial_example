cmake_minimum_required(VERSION 3.5)
project(cuda_tutorial_example CUDA)

#set(CMAKE_CUDA_ARCHITECTURES 75)# T4
set(CMAKE_CUDA_ARCHITECTURES 80)# A100

set(CMAKE_CUDA_STANDARD 11)
include_directories(${PROJECT_SOURCE_DIR})

#add_subdirectory(add)

add_subdirectory(dot)

#add_subdirectory(transpose)

#add_subdirectory(sgemm)


