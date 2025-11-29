#!/bin/bash

git clone https://github.com/NVIDIA/cuda-samples.git --recursive

cd ./cuda-samples/Samples/1_Utilities/deviceQuery || exit

rm -rf ./build
mkdir -p build
cd ./build || exit
cmake ..
make

sudo cp deviceQuery /usr/local/bin/

deviceQuery

