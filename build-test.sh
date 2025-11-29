#!/bin/bash
rm -rf ./build
mkdir -p build
cd ./build || exit
cmake .. > /dev/null 2>&1
make

execute(){
  executable_path="./$1/$1"
  if [ -f "$executable_path" ]; then
    "$executable_path"
  else
    echo "$executable_path 不存在\n"
  fi
}

execute add
execute dot
execute transpose
execute sgemm