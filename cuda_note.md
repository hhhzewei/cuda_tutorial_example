## api
`cudaMallocManaged`: 统一内存，cpu和gpu都可以访问，实际是通过页错误迁移

`cudaDeviceSynchronize`: 同步cuda设备，等核函数执行结束

`cudaFree`：释放gpu内存

`cudaMemPrefetchAsync`：提前把数据放到gpu

`cudaMalloc`：在gpu上分配内存

`cudaMemcpy`：在cpu和gpu之间转移内存数据

`cudaGetLastError`

`float4`：按4个float长度访存

## 优化方向
### elementwise
1.strip loop，每个线程处理对应的元素。
- 全局内存读2N，写N。

2.如果需要四轮以上迭代，用float4访存减少访问全局内存。
- 全局内存读$2N\times 1/4=N/2$，写$N\times 1/4=N/4$。

### reduce
1.原地半程迭代
- 全局内存，读$N+N/2+N/4...+2=2N-2$，写$N/2+N/4+N/8...+1=N-1$。

2.block内半程迭代，再atomic相加。
- 全局内存读N，写blockNum。
- 共享内存读$blockNum\times(threadNum+threadNum/2+...+2)+blockNum=2N-blockNum$，写$N+blockNum\times(threadNum/2+threadNum/4+...+1)=2N-blockNum$。

3.先warp内shuffle，每warp结果转移到共享内存，再转移到首warp内shuffle，获得block内规约结果，最后各block进行atomic合并。

### transpose
1. 线程(x,y)读取`in[y][x]`写入`out[x][y]`。
   - warp读取全局内存连续（coalescing），写入不连续。
2. 线程(x,y)读`in[y][x]`写入warpSize * warpSize共享内存`tile[ty][tx]`，再读`tile[tx][ty]`写入`out[x0+ty][y0+tx]`。
   - 同warp写共享内存不同bank，但是读相同bank，存在bank conflict。
3. padding：在2基础上共享内存尺寸改为warpSize * (warpSize+1)。
   - 每一行加一格，让每一行同列元素bank号错开。
4. swizzle：线程(x,y)读`in[y][x]`写入warpSize * warpSize共享内存`tile[ty][tx^ty]`，再读`tile[tx][ty^tx]`写入`out[x0+ty][y0+tx]`。
   - 本质是在寻找一个拉丁方。
   - 解决bank conflict，同时写入全局内存coalesce 


### Shared Memory Bank 的基本原理

- Shared memory 被划分成 banks（通常是 32 个 banks，对现代 NVIDIA GPU 来说每个 bank 宽度是 4 字节或者 8 字节，取决于数据类型）。

- 每个 warp（32 个线程）访问 shared memory 时，如果访问的地址落在不同的 bank，那么可以 同时并行访问。

- 如果同一个 warp 中多个线程访问了同一个 bank 的不同地址，就会发生 bank conflict，需要被序列化访问，性能下降。

- 同一个 warp 中，每个线程访问同一 bank 的同一地址（broadcast） 是特殊情况，硬件会优化，只需一次访问。