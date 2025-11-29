[文档地址](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#)

[APOD设计周期](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#assess-parallelize-optimize-deploy)：Assess, Parallelize, Optimize, Deploy

加速比（speedup）：单核心执行时间/当前执行时间

线性加速：处理器数增加为N倍，加速比也增加为N倍

## [扩展性（scaling）](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#understanding-scaling)

### 强扩展性（（Strong Scaling）和阿姆达尔定律（Amdahl's Law）

- 假设**问题规模固定**，串行执行时间中可并行化部分占比为P， N是并行代码部分运行的处理器数，那么加速比S为

$$$
S=\frac{1}{1-P+\frac{P}{N}}
$$$

- 即便处理器核心无限多，加速比上限为$\frac{1}{1-P}$

### 弱扩展性（Weak Scaling）和古斯塔夫森定律（Gustafson's Law）

- 假设问题并行部分规模随处理器数增多，加速比为
  $$$
  S=\frac{1-p+N*p}{1-p+\frac{N*p}{N}}=1-p+Np
  $$$
- 如果在增加处理器数的同时，扩大问题规模，加速比（处理器利用率）可以获得提升

## [Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#memory-optimizations)

### host memory

#### 分页内存（Pageable Memory）（OS前置知识）

每个进程都拥有一个完整的虚拟内存空间，其远大于实际的物理内存空间。

进程在`malloc`时实际只分配了虚拟内存，获得虚拟内存地址，只有真正初次使用内存时才会触发page fault，去分配真正的物理内存页面，并将二者地址映射记录在页表中。

当物理内存不足时，os会将不活跃的页面移入磁盘的swap文件中，在页表中记录下这个映射以及页面状态，再次访问时触发page fault移回。

Out Of Memory错误实际是**物理内存+swap文件**不足。

#### 固定内存（Pinned Memory）

```c++
// cudaMallocHost() // 旧的分配方法，等价于默认flag的cudaHostAlloc
cudaHostAlloc()// 分配固定内存
cudaHostRegister() // 固定已分配的host内存
```

- 固定内存是一种host内存，区别于一般的分页（pageable）内存。
- `cudaHostAlloc`会确保物理内存被分配，且不可被交换（锁页）。
- 固定内存可实现host与device之间的**最高带宽**
- 因为物理内存有限，所以固定内存是一种**稀缺**资源
- 分配开销很**重**，远大于普通的`malloc`
- 可用于数据传输或者device直接访问host内存

#### stream

```c++
cudaStream_t stream;
cudaStreamCreate(&stream);
```

- 默认流**0**的上的所有操作（内存分配、复制、核函数调用）只有在device上所有流中的所有先前调用完成后才会开始
- 一般使用非默认流执行异步操作确保并行。

#### 异步数据传输

```c++
// 异步传输数据和调用核函数
cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, 0);
kernel<<<grid, block>>>(a_d);
cpuFunction();
// 分批传输数据
size=N*sizeof(float)/nStreams;
for (i=0; i<nStreams; i++) {
    offset = i*N/nStreams;
    cudaMemcpyAsync(a_d+offset, a_h+offset, size, dir, stream[i]);
    kernel<<<N/(nThreads*nStreams), nThreads, 0,
             stream[i]>>>(a_d+offset);
}
```

- `cudaDeviceProp`的`asyncEngineCount`指示内核执行与主机设备间数据传输是否能重叠执行
- `cudaMemcpyAsync`传输的主机内存需要固定内存，否则会创建固定内存拷贝一次再传输，不是真正的异步调用。

#### 零拷贝（Zero Copy）

```c++
float *a_h, *a_map;
...
// 检查设备是否支持将主机内存映射到设备的地址空间
cudaGetDeviceProperties(&prop, 0);
if (!prop.canMapHostMemory)
    exit(0);
// 启用页面锁定内存映射
cudaSetDeviceFlags(cudaDeviceMapHost);
// flag为cudaHostAllocMapped
cudaHostAlloc(&a_h, nBytes, cudaHostAllocMapped);
cudaHostGetDevicePointer(&a_map, a_h, 0);
kernel<<<gridSize, blockSize>>>(a_map);
```

- 零拷贝允许 GPU 线程直接访问主机内存。
- 在集成GPU（integrated为1）上，映射固定内存始终能提升性能，因为host和device共享物理内存。
- 在独立 GPU 上，映射固定内存仅在某些情况下具有优势，因为要通过**PCIE**（peripheral component interconnect express）总线访问，延迟超大，带宽小。

#### 统一虚拟寻址（Unified Virtual Addressing）

### Device Memory

#### Global Memory

SM <- L1 cache(片上SRAM) <- L2 cache(片上SRAM) <- DRAM

线程warp访问的全局内存连续且对齐，确保可以合并（coalesce）为事务
- 计算能力6.0之前，如果开启L1 cache（默认），事务大小为128B（L1 Cache Line），否则为32B。
- 6.0之后，L1从全局内存中取消，事务大小固定为32B。

#### Shared Memory

共享内存被划分为大小相等的bank，对不同bank内存的访问可以同时进行。

不同线程对同一bank访问会发生bank conflict串行访问，导致带宽降低。

不过warp的多个线程访问同一个地址（显然同bank）时会发生**广播**，来自不同bank的广播会被合并成一个**多播**。

bank在每个cycle具有**32位带宽**，并且连续的**32位字**被分配给bank。线程束大小为32个线程，bank数也为32。

###### 异步load
CUDA 11.0引入异步搬移global memory数据到shared memory的能力。
```c++
template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count) {
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  //pipeline pipe;
  for (size_t i = 0; i < copy_count; ++i) {
    __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                            &global[blockDim.x * i + threadIdx.x], sizeof(T));
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);

  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}
```
- 同步取全局内存赋值共享内存，实际先存放到中间寄存器，存在额外两条指令，异步则节省两条指令。
- 计算和load数据可以同时进行。
- 本质线程只提交load任务，真正是由copy engine硬件执行load。

#### Local Memory(局部内存)

位于片外DRAM

#### Texture Memory(纹理内存)

#### Constant Memory(常量内存 )

#### Registers(寄存器)

访问寄存器几乎消耗0 cycle

### Occupancy(占用率)

占用率是每个SM上活跃warp数与最大活跃warp数之比

占用率的一个因素是寄存器的可用性：SM上寄存器数量有限，限制了SM上活跃的线程块数