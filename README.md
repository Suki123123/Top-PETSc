# 3D Topology Optimization with GPU Acceleration

基于PETSc的3D拓扑优化程序，支持GPU加速和多GPU并行计算。

## 特性

- **GPU加速**: 使用CUDA加速矩阵运算和求解器
- **多GPU支持**: 支持单GPU和双GPU并行计算
- **Matrix-Free**: 无矩阵模式，节省内存
- **PCG求解器**: 预条件共轭梯度法 (CG + Jacobi)
- **OC优化器**: Optimality Criteria优化算法
- **PDE滤波器**: 基于PDE的密度滤波
- **简化参数**: 固定常用参数，简化使用

## 固定配置

程序已将常用参数固定在代码中：

- **体积分数**: 0.12 (12%)
- **惩罚因子**: 3.0 (SIMP)
- **滤波器**: PDE滤波器，rmin自动计算（1.5倍单元尺寸）
- **求解器**: PCG (CG + Jacobi)
- **KSP最大迭代**: 1000
- **收敛标准**: ch < 0.02
- **输出频率**: 每10次迭代输出一次

## 系统要求

- **操作系统**: Linux
- **编译器**: GCC/G++ with CUDA support
- **依赖库**:
  - PETSc 3.21.0 (with CUDA support)
  - CUDA 12.2+
  - OpenMPI (with CUDA-aware support)
  - Python 3 (用于VTK转换)

## 编译

```bash
cd src
make
```

## 使用方法

### 基本用法

```bash
# 单GPU
./src/topopt -nx 65 -ny 33 -nz 33

# 双GPU
mpirun -np 2 ./src/topopt -nx 65 -ny 33 -nz 33
```

### 可选参数

- `-nx <数值>`: X方向节点数（必需）
- `-ny <数值>`: Y方向节点数（必需）
- `-nz <数值>`: Z方向节点数（必需）
- `-maxItr <数值>`: 最大迭代次数（默认200）
- `-output_final_vtk`: 输出最终VTK文件

### 示例

```bash
# 双GPU，200次迭代，输出VTK
mpirun -np 2 ./src/topopt -nx 65 -ny 33 -nz 33 -maxItr 200 -output_final_vtk
```

## VTK可视化

生成VTK文件后，使用Python脚本转换为ParaView格式：

```bash
cd src
python3 bin2vtu_v3.py 0
```

生成的`.vtu`文件可以用ParaView打开查看。

## 输出说明

程序每10次迭代输出一次，包含以下信息：

- **It.**: 迭代次数
- **真实fx**: 真实目标函数值
- **缩放fx**: 缩放后的目标函数值
- **gx[0]**: 体积约束值
- **ch.**: 设计变化（收敛指标）
- **mnd.**: 离散度量
- **KSP**: 本次KSP迭代次数
- **总KSP**: 累计KSP迭代次数
- **time**: 本次迭代时间（秒）

## 配置文件

`configs/options_pcg_standard.txt` - PCG求解器标准配置（可选使用）

## 项目结构

```
.
├── src/                    # 源代码
│   ├── main.cc            # 主程序
│   ├── TopOpt.cc/h        # 拓扑优化参数
│   ├── LinearElasticity.cc/h  # 线性弹性求解器
│   ├── OC.cc/h            # OC优化器
│   ├── Filter.cc/h        # 滤波器基类
│   ├── PDEFilter.cc/h     # PDE滤波器
│   ├── MPIIO.cc/h         # VTK输出
│   ├── MatrixFreeGPU.cu/h # GPU Matrix-Free实现
│   ├── bin2vtu_v3.py      # VTK转换脚本
│   └── makefile           # 编译配置
├── configs/               # 配置文件
│   └── options_pcg_standard.txt
└── README.md             # 本文件
```

## 注意事项

1. **收敛性**: 固定penal=3.0可能导致收敛困难，建议使用延拓策略
2. **内存**: Matrix-Free模式节省内存，适合大规模问题
3. **GPU**: 确保CUDA_VISIBLE_DEVICES正确设置
4. **KSP迭代**: 如果KSP经常达到最大迭代次数，考虑增加maxiter或使用更强的预条件器

## 许可证

本项目基于原始TopOpt_in_PETSc项目修改。

## 作者

原始项目: Niels Aage, Erik Andreassen, Boyan Lazarov (2013-2019)

GPU加速和简化版本修改: 2026

## 参考文献

- Aage, N., Andreassen, E., Lazarov, B. S., & Sigmund, O. (2017). Giga-voxel computational morphogenesis for structural design. Nature, 550(7674), 84-86.
