#!/bin/bash

nsys_execute(){
  executable_path="./$1/$1"
  if [ -f "$executable_path" ]; then
    nsys profile -o "$1" --trace=cuda "$executable_path"
    nsys stats --report cuda_gpu_kern_sum --force-overwrite true "$1".nsys-rep
  else
    echo "$executable_path 不存在\n"
  fi
}

nsys_execute add
nsys_execute dot
nsys_execute transpose
nsys_execute sgemm
