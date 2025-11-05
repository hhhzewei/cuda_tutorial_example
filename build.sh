#!/bin/bash

export PATH=/opt/nvidia/nsight-systems/2025.5.1/bin:$PATH
rm -rf ./build
mkdir -p build
cd ./build || exit
cmake .. > /dev/null 2>&1
make
nsys_execute1(){
  nsys profile -t cuda --stats=true ./"$1"
}
nsys_execute2(){
  executable_path="./$1"
  if [ -f "$executable_path" ]; then
    nsys profile -o "$1" --trace=cuda "$executable_path"
    nsys stats --report cuda_gpu_kern_sum --force-overwrite true "$1".nsys-rep
  else
    echo "$executable_path 不存在\n"
  fi
}
#nsys_execute2 add
#nsys_execute2 dot
#nsys_execute2 transpose
nsys_execute2 sgemm
