## RUN
在[colab](https://colab.research.google.com/)上跑，注意使用终端而不是记事本，否则安装nsight软件的用户协议会出问题。
1. 运行prepare

    把所有脚本文件直接复制到`/content`(colab默认目录下)，然后运行[prepare](./prepare.sh)脚本安装nsys，ncu默认有不用安装，前者获得概要信息，后者获得核函数执行详细信息。

    ```bash
    # 要修改PATH所以用source
    source ./prepare.sh
    ```
2. 构建运行
   
   把源代码和cmake文件复制到对应位置，执行[nsys构建运行](./build-nsys.sh)或者[ncu构建运行](./build-ncu.sh)

   ```bash
   
   ```