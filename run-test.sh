#!/bin/bash

cd build || exit

execute(){
  executable_path="./$1/$1"
  if [ -f "$executable_path" ]; then
    "$executable_path"
  else
    echo "$executable_path 不存在\n"
  fi
}

#execute add
#execute dot
#execute transpose
#execute sgemm
execute softmax
#execute hgemm
