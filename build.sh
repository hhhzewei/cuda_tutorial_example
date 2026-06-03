#!/bin/bash

rm -rf ./build
mkdir -p build
cd ./build || exit
# 获取云端 libtorch位置
TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="${TORCH_PATH}"/Torch .. #> /dev/null 2>&1
make
