#! /bin/bash

# 定义常量
NSYS_RUNFILE="NsightSystems-linux-public-2025.5.1.121-3638078.run"
INSTALL_DIR="/opt/nvidia/nsight-systems/2025.5.1/bin"

# 下载文件
if [ ! -f "$NSYS_RUNFILE" ]; then
    echo "文件不存在，开始下载..."
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_5/$NSYS_RUNFILE
else
    echo "文件已存在，跳过下载"
fi
# 修改权限
chmod +x $NSYS_RUNFILE
# 安装到 /usr/local/nsys
sudo ./$NSYS_RUNFILE
# 加入 PATH
export PATH="$INSTALL_DIR"/bin:$PATH

# 验证
nsys --version