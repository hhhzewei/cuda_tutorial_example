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

新版改成8\*16block（线程数没变），显然读a的shared tile的横向条带，每一轮同warp读相同地址，可以合并；而读b的shared tile的纵向条带，因为相邻线程读取地址隔了个THREAD_TILE_N，所以32个线程的bank序列大致为0,2,4,...,30,0,2,4,...,30。

拓展一下就是假设THREAD_TILE_N=k，那么bank序列肯定就是，0,k,2k,...mk,...这么重复。

设warp_circle=warp_size/k，那么tx/warp_circle表示在重复的第几个circle，很显然根据第几个circle作为offset就行了。

以上两点看起来很完美，结果跑出来结果很吓人，时间反而变成4倍多，甚至还不如naive版本，纯纯负优化。

问gpt也是百思不得其解，不过偶然发现自己sgemm的共享内存类型都写成unsigned了，虽然很逆天，不过其实不影响太多，而且算子相对性能肯定没影响。

最后把offset直接删了，没想到直接就解决了，尽管仍然是负优化，但是差得不多。

然后听gpt的把除法、取余都用位运算，最后的最后，结果还是惊天负优化，明天再想吧

## 2025/11/8

今天尝试解决负优化的v1版本，先是尽量把offset的计算前置转成编译期常量，然后把除法取余都换成位运算，还是解决不了。

实验了下把维度从8\*32换成16\*16，这样offset总为0，bank访问逻辑和v0相同，然而性能还是没回去。

结合之前实验去掉offset逻辑负优化就解除了，大概就是循环里offset计算开销导致的。

再往后写了个v2使用16\*16的block，然后tile用padding，解决访问tile b的bank conflict，负优化没了，不过还是和v0版本比差一点点。

不过想到可能还有coalesce等问题，准备再写个优化点版本，总之v0版本直觉就有很多问题，不知道为什么一直超越不了。

v0问题一是读全局内存，每个线程是读对应2*2的thread tile，按理说warp就没coalesce；二是访问tile b也是有bank conflict，比如第一轮，线程\[0\]\[0~15\]和\[1\]\[0~15\]同属一个warp，但是bank序列相同没散开。
