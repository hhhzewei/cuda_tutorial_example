#!/bin/bash
ncu --version
rm -rf ./build
mkdir -p build
cd ./build || exit
# 获取云端 libtorch位置
TORCH_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake -DCMAKE_PREFIX_PATH="${TORCH_PATH}"/Torch .. #> /dev/null 2>&1
make

ncu_execute(){
  executable_path="./$1/$1"
  if [ -f "$executable_path" ]; then
#    ncu  -o "$1" "$executable_path" # 显示概要信息
    ncu --set full -o "$1" "$executable_path" # 显示完整信息
  else
    echo "$executable_path 不存在\n"
  fi
}

ncu_execute add
ncu_execute dot
ncu_execute transpose
ncu_execute sgemm
ncu_execute softmax
