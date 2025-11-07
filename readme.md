

## RUN
在colab上跑

1. 检查[CmakeLists文件](./CMakeLists.txt)，`CMAKE_CUDA_ARCHITECTURES`属性务必和GPU型号一致，如T4为75，A100为80，否则编译结果无法运行。

2. 提供三种构建运行脚本
   - [build-test](./build-test.sh)测试算子跑通。
   - [build-nsys](./build-nsys.sh)使用nsys工具分析算子，生成报告，同时打印概要信息，主要是运行时间。
      - colab默认没有nsys，要提前安装。运行[安装脚本](./nsys-install.sh)即可，注意不要在jupyter而是在终端运行，否则安装时的用户交互界面会出错。
     ```shell
       # source确保PATH修改生效
       source ./nsys-install.sh
      ```
   - [build-ncu](./build-nsys.sh)使用ncu工具分析算子，生成详细信息报告，下载到本地用Nsight Compute软件打开
     - colab默认有ncu工具，不必重复安装，而且尝试安装最新版反而运行失败。
