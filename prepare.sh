#!/bin/bash

# 安装vcpkg
# 解压源码
rm -fr ./project
mkdir -p project
tar -xzvf project.tar.gz -C ./project
source ./project/vcpkg-install.sh
#source ./nsys-install.sh
# colab默认安装ncu，版本完全足够
