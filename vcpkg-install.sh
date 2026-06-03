#!/bin/bash

git clone https://github.com/microsoft/vcpkg.git
cd vcpkg || exit
./bootstrap-vcpkg.sh
VCPKG_ROOT="$(pwd)"
export VCPKG_ROOT=${VCPKG_ROOT}
export PATH=$VCPKG_ROOT:$PATH
cd ..
vcpkg install nvidia-cutlass

