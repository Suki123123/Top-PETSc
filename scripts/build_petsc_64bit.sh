#!/bin/bash
#=============================================================================
# PETSc 64位索引自动配置和编译脚本
#=============================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始配置和编译64位索引PETSc"
echo "预计总时间: 40-70分钟"
echo "=========================================="
echo ""

# 设置环境变量
export PETSC_DIR=/home/user/petsc-cuda/petsc-3.21.0
export PETSC_ARCH=arch-cuda-mpi-64

# 步骤1: 配置PETSc
echo "[1/4] 配置PETSc (64位索引)..."
echo "预计时间: 5-10分钟"
cd $PETSC_DIR

python3 ./configure \
  PETSC_ARCH=$PETSC_ARCH \
  --with-mpi-dir=$HOME/openmpi-cuda \
  --with-fc=0 \
  --with-cuda=1 \
  --with-cuda-dir=/usr/local/cuda-12.2 \
  --with-cudac=nvcc \
  --with-cuda-arch=89 \
  --with-64-bit-indices=1 \
  --with-debugging=0 \
  --with-shared-libraries=1 \
  --with-x=0 \
  --download-hypre=1 \
  --download-metis=1 \
  --download-parmetis=1 \
  COPTFLAGS=-O3 \
  CXXOPTFLAGS=-O3

echo ""
echo "[1/4] 配置完成 ✓"
echo ""

# 步骤2: 编译PETSc
echo "[2/4] 编译PETSc..."
echo "预计时间: 30-60分钟"
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all

echo ""
echo "[2/4] 编译完成 ✓"
echo ""

# 步骤3: 验证PETSc
echo "[3/4] 验证PETSc..."
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check

echo ""
echo "[3/4] 验证完成 ✓"
echo ""

# 步骤4: 重新编译topopt
echo "[4/4] 重新编译topopt..."
cd /home/user/sq_cuda/Top-in-PETSc/TopOpt_in_PETSc

# 备份原makefile
cp makefile makefile.32bit.bak

# 更新PETSC_ARCH
sed -i 's/PETSC_ARCH=arch-cuda-mpi$/PETSC_ARCH=arch-cuda-mpi-64/' makefile

# 清理并重新编译
make clean
make

echo ""
echo "[4/4] topopt编译完成 ✓"
echo ""

echo "=========================================="
echo "全部完成！"
echo "=========================================="
echo ""
echo "现在可以测试大规模问题:"
echo ""
echo "单GPU测试 (129×129×65):"
echo "  TopOpt_in_PETSc/topopt -options_file options_gpu_balanced.txt -nx 129 -ny 129 -nz 65 -nlvls 4 -maxItr 3"
echo ""
echo "双GPU测试 (129×129×65):"
echo "  export OPENMPI_CUDA_HOME=\$HOME/openmpi-cuda"
echo "  export PATH=\$OPENMPI_CUDA_HOME/bin:\$PATH"
echo "  export LD_LIBRARY_PATH=\$OPENMPI_CUDA_HOME/lib:\$LD_LIBRARY_PATH"
echo "  mpirun -np 2 TopOpt_in_PETSc/topopt -options_file options_dual_gpu_production.txt -nx 129 -ny 129 -nz 65 -nlvls 4 -maxItr 3"
echo ""
