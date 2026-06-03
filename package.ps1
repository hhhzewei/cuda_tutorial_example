Remove-Item -Force ./project.tar.gz
tar -czvf project.tar.gz `
./add  ./dot ./element_wise ./reduce ./sgemm ./hgemm ./softmax ./transpose ./util ./CMakeLists.txt `
./vcpkg-install.sh `
./build.sh ./build-test.sh ./run-ncu.sh ./run-test.sh