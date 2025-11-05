#!/bin/bash

# 定义常量
URL="https://developer.download.nvidia.cn/assets/tools/secure/nsight-compute/2025_3_1/nsight-compute-linux-2025.3.1.4-36398880.run?__token__=exp=1762346440~hmac=08a57233e0ace8133f8c6c955ddd4a3569ca610d32e1c0a266d455512e63d7ad&t=eyJscyI6ImdzZW8iLCJsc2QiOiJodHRwczovL3d3dy5nb29nbGUuY29tLmhrLyJ9"
NCU_RUNFILE="nsight-compute.run"
INSTALL_DIR="/usr/local/NVIDIA-Nsight-Compute-2025.3"

# 下载文件
if [ ! -f "$NCU_RUNFILE" ]; then
    echo "文件不存在，开始下载..."
wget "$URL" -O "$NCU_RUNFILE"
else
    echo "文件已存在，跳过下载"
fi
# 修改权限
chmod +x "$NCU_RUNFILE"
# 安装到默认路径
sudo ./"$NCU_RUNFILE"
# 加入 PATH
export PATH="$INSTALL_DIR":$PATH
# 验证
ncu --version