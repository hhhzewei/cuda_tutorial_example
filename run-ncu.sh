#!/bin/bash

ncu --version

cd build || exit

ncu_execute(){
  executable_path="./$1/$1"
  if [ -f "$executable_path" ]; then
#    ncu  -o "$1" "$executable_path" # 显示概要信息
    ncu --set full --import-source yes --cache-control all --clock-control base -o "$1" "$executable_path" # 显示完整信息
  else
    echo "$executable_path 不存在\n"
  fi
}

ncu_execute2(){
  dir="./$1"
  app="$2"
  executable_path="${dir}/${app}"
  if [ -f "$executable_path" ]; then
    ncu --set full --import-source yes --cache-control all --clock-control base -k regex:".*$3.*" -o "$3" "$executable_path" # 显示完整信息
  else
    echo "$executable_path 不存在\n"
  fi
}

#ncu_execute add
#ncu_execute dot
#ncu_execute transpose
#ncu_execute softmax
# sgemm
#ncu_execute2 sgemm sgemm ampere_sgemm
#ncu_execute2 sgemm sgemm sgemm_naive
#ncu_execute2 sgemm sgemm sgemm_block_tile
#ncu_execute2 sgemm sgemm sgemm_thread_tile_v0
#ncu_execute2 sgemm sgemm sgemm_thread_tile_v2
#ncu_execute2 sgemm sgemm sgemm_thread_tile_v3
#ncu_execute2 sgemm sgemm sgemm_thread_tile_v4
#ncu_execute2 sgemm sgemm sgemm_thread_tile_v5
#ncu_execute2 sgemm sgemm sgemm_thread_tile_v6
#ncu_execute2 sgemm sgemm sgemm_thread_tile_v7
#ncu_execute2 sgemm sgemm sgemm_thread_tile_v8
#ncu_execute2 sgemm sgemm sgemm_tensor_core_v1
#ncu_execute2 sgemm sgemm sgemm_tensor_core_v2
#ncu_execute2 sgemm sgemm gemm_sm80_cuda
# hgemm
#ncu_execute2 hgemm hgemm hgemm_sm80_cute
#ncu_execute2 hgemm hgemm gemm_device_cute_example
#ncu_execute2 hgemm hgemm sm80_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize160x128x32_stage4_warpsize2x2x1_tensor16x8x16_execute_kernel__5x_cublas
#ncu_execute2 hgemm hgemm Kernel
ncu_execute hgemm