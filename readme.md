## RUN
在[colab](https://colab.research.google.com/)上跑，注意使用终端而不是记事本，否则安装nsight软件的用户协议会出问题。
0. pre-pre

    Nsight System可以直接`wget`安装，但是Nsight Compute下载还要登录Nvidia账号，`wget`的链接要时限动态参数。所以先到[Nsight Compute官网](https://developer.nvidia.com/tools-overview/nsight-compute/get-started)手动下载，在浏览器下载栏复制下载URL到[ncu安装脚本](./ncu-install.sh)。

1. 运行prepare

    把所有脚本文件直接复制到`/content`(colab默认目录下)，然后运行[prepare](./prepare.sh)脚本安装nsys和ncu分析工具，前者获得概要信息，后者获得核函数执行详细信息。

    ```bash
    # 要修改PATH所以用source
    source ./prepare.sh
    ```
2. 构建运行
   
   把源代码和cmake文件复制到对应位置，执行[nsys构建运行](./build-nsys.sh)或者[ncu构建运行](./build-ncu.sh)

   ```bash
   
   ```