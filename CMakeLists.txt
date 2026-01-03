cmake_minimum_required(VERSION 3.5)
project(cuda_tutorial_example CXX CUDA)

#set(CMAKE_CUDA_ARCHITECTURES 75)# T4
set(CMAKE_CUDA_ARCHITECTURES 80)# A100

# colab上torch需要17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_PREFIX_PATH "D:/CppDevelop/Lib/libtorch") # 追加，避免覆盖build脚本里-D选项

find_package(Torch REQUIRED)

#find_package(NvidiaCutlass CONFIG REQUIRED) # vcpkg安装完成提示

include_directories(${PROJECT_SOURCE_DIR})

add_subdirectory(add)

add_subdirectory(dot)

add_subdirectory(transpose)

#add_subdirectory(sgemm)

add_subdirectory(softmax)

