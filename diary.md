# 2025/11/7
今天决定开始记录一些关于进展的碎碎念。

先说下昨天，昨天是想优化下thread tile脚本，不过先在kernel层上加了个call层，统一host 和 device 间的内存操作，然后测试的时候偶然发现dot swizzle打印的结果是错的，自己一直关注运行时间没发现。

中间加个打印语句，发现计算结果是正确结果的1/8，检查了很久才解决，详情看git修订d457f8ff99418b0dd8e85ccef72f1aa8cee45a65。

两个逆天错误，一是计算laneId，threadIdx写成blockDim，二是block内用首warp合并各warp结果时，`+=`写成了`=`，结果只有一个warp
，正好一个block有8个warp。

后面因为colab额度用完就没继续，打游戏了。

今天主要就优化sgemm的thread tile版本，基本就两点。

一：旧版是一个线程读全局内存只读自己thread tile对应那部分对应的小tile，会导致warp不coalesce，改成读全局内存到共享内存用二维的strip loop。

二：旧版是16\*16的block，线程读b的shared tile的一个纵向条带到寄存器，会很显然第2i、2i+1行线程构成warp，但是二者的bank序列相同。

新版改成32\*8block（线程数没变），显然读a的shared tile的横向条带，每一轮同warp读相同地址，可以合并；而读b的shared tile的纵向条带，因为相邻线程读取地址隔了个THREAD_TILE_N，所以32个线程的bank序列大致为0,2,4,...,30,0,2,4,...,30。

拓展一下就是假设THREAD_TILE_N=k，那么bank序列肯定就是，0,k,2k,...mk,...这么重复。

设warp_circle=warp_size/k，那么tx/warp_circle表示在重复的第几个circle，很显然根据第几个circle作为offset就行了。

以上两点看起来很完美，结果跑出来结果很吓人，时间反而变成4倍多，甚至还不如naive版本，纯纯负优化。

问gpt也是百思不得其解，不过偶然发现自己sgemm的共享内存类型都写成unsigned了，虽然很逆天，不过其实不影响太多，而且算子相对性能肯定没影响。

最后把offset直接删了，没想到直接就解决了，尽管仍然是负优化，但是差得不多。

然后听gpt的把除法、取余都用位运算，最后的最后，结果还是惊天负优化，明天再想吧
