# TopOpt GPU加速项目

PETSc拓扑优化程序的GPU加速实现，支持单GPU和双GPU并行计算。

## 快速开始

### 单GPU运行
```bash
TopOpt_in_PETSc/topopt \
    -options_file options_gpu_balanced.txt \
    -nx 65 -ny 33 -nz 33 -nlvls 4 -maxItr 100
```

### 双GPU运行
```bash
# 设置环境变量
export OPENMPI_CUDA_HOME=$HOME/openmpi-cuda
export PATH=$OPENMPI_CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$OPENMPI_CUDA_HOME/lib:$LD_LIBRARY_PATH

# 运行
mpirun -np 2 TopOpt_in_PETSc/topopt \
    -options_file options_dual_gpu_production.txt \
    -nx 65 -ny 65 -nz 33 -nlvls 4 -maxItr 100
```

## 配置文件

- **options_gpu_balanced.txt** - 单GPU最优配置（推荐）
- **options_dual_gpu_production.txt** - 双GPU生产配置
- **options_gpu.txt** - 基础GPU配置

## 性能对比

### 小规模问题 (65×33×33, 3次迭代)

| 配置 | 平均时间 | 加速比 |
|------|---------|--------|
| CPU | 8.02秒 | 1.0× |
| 单GPU | 0.95秒 | 8.44× |
| 双GPU | 1.43秒 | 5.61× |

### 大规模问题 (129×65×65, 3次迭代)

| 配置 | 平均时间 | 加速比 |
|------|---------|--------|
| 单GPU | 7.72秒 | 1.0× |
| 双GPU | 4.48秒 | 1.72× |

**结论**: 
- 小规模问题: 单GPU最优 (通信开销大于并行收益)
- 大规模问题: 双GPU最优 (并行收益超过通信开销)
- 问题规模越大，双GPU优势越明显

## 网格尺寸要求

使用多层网格(nlvls>1)时，网格尺寸必须满足：
```
(nodes-1) 能被 2^(nlvls-1) 整除
```

nlvls=4时，有效节点数: 9, 17, 25, 33, 65, 129

示例:
- ✓ 65×65×33
- ✓ 65×33×33  
- ✗ 65×64×64 (64不满足要求)

## 规模限制

当前配置(32位PETSc索引)的最大问题规模:
- 单GPU: 约130×130×65 (受GPU内存限制，24GB)
- 双GPU: 约130×65×65 (受32位整数索引限制)

更大规模需要重新编译PETSc使用64位索引 (--with-64-bit-indices)

## 系统配置

- **PETSc**: 3.21.0 (arch-cuda-mpi)
- **CUDA**: 12.2
- **OpenMPI**: 4.1.6 (GPU-aware)
- **GPU**: NVIDIA RTX 4090 × 2

## 详细文档

- **双GPU使用说明.txt** - 双GPU配置详细说明和性能数据
