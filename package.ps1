Remove-Item -Force ./project.tar.gz
tar -czvf project.tar.gz `
./add  ./dot ./element_wise ./reduce ./sgemm ./softmax ./transpose ./util ./CMakeLists.txt ./build-ncu.sh