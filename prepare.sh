#!/bin/bash
rm -fr ./project
mkdir -p project
tar -xzvf project.tar.gz -C ./project
cd ./project
sh build-ncu.sh
#source ./nsys-install.sh
# colab默认安装ncu，版本完全足够
