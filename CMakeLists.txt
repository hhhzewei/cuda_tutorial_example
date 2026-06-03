cmake_minimum_required(VERSION 3.30)
# 必须在project之前
set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")

project(cuda_tutorial_example CXX CUDA)

# colab上torch需要17
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CUDA_ARCHITECTURES 75)# T4
set(CMAKE_CUDA_ARCHITECTURES 80)# A100

# 默认release模式构建
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_PREFIX_PATH "D:/CppDevelop/Lib/libtorch/") # 追加，避免覆盖build脚本里-D选项

find_package(Torch REQUIRED)

find_package(NvidiaCutlass CONFIG REQUIRED) # vcpkg安装完成提示

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/cutlass/include)


# 检查是否使用的是 MSVC 编译器
if(MSVC)
    # 1. 明确指定：只有在编译纯 C++ 语言（CXX）时，才直接加 /Zc:preprocessor
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>)

    # 2. 明确指定：在编译 CUDA 语言（.cu）时，必须加上 -Xcompiler 前缀包裹
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

#add_subdirectory(add)
#
#add_subdirectory(dot)
#
#add_subdirectory(transpose)

add_subdirectory(sgemm)

add_subdirectory(hgemm)

add_subdirectory(softmax)

