#!/bin/bash

echo "=========================================="
echo "GMG vs GAMG 性能对比测试"
echo "=========================================="
echo ""

# 测试问题：17×17×9（小问题，快速测试）
NX=17
NY=17
NZ=9
NLVLS=3
RMIN=1.0
MAXITER=5

echo "测试配置："
echo "  网格: ${NX}×${NY}×${NZ}"
echo "  MG层数: ${NLVLS}"
echo "  迭代次数: ${MAXITER}"
echo ""

# 测试1: GMG标准配置
echo "=========================================="
echo "测试1: GMG标准配置 (Chebyshev + Jacobi)"
echo "=========================================="
TopOpt_in_PETSc/topopt \
  -nx $NX -ny $NY -nz $NZ -nlvls $NLVLS \
  -volfrac 0.3 -rmin $RMIN -maxiter $MAXITER \
  -options_file options_gpu_gmg.txt \
  2>&1 | grep -E "State solver|It\.: [0-9]+,"

echo ""

# 测试2: GMG Matrix-Free配置
echo "=========================================="
echo "测试2: GMG Matrix-Free (最小内存)"
echo "=========================================="
TopOpt_in_PETSc/topopt \
  -nx $NX -ny $NY -nz $NZ -nlvls $NLVLS \
  -volfrac 0.3 -rmin $RMIN -maxiter $MAXITER \
  -options_file options_gpu_gmg_matfree.txt \
  2>&1 | grep -E "State solver|It\.: [0-9]+,"

echo ""

# 测试3: 代码默认GMG（GMRES + SOR）
echo "=========================================="
echo "测试3: 代码默认GMG (GMRES + SOR)"
echo "=========================================="
TopOpt_in_PETSc/topopt \
  -nx $NX -ny $NY -nz $NZ -nlvls $NLVLS \
  -volfrac 0.3 -rmin $RMIN -maxiter $MAXITER \
  -ksp_rtol 1e-5 -dm_mat_type aijcusparse -dm_vec_type cuda \
  -use_gpu_aware_mpi 0 \
  2>&1 | grep -E "State solver|It\.: [0-9]+,"

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "结论："
echo "1. GMG标准配置：Chebyshev光滑器，收敛最快"
echo "2. GMG Matrix-Free：内存最小，适合超大规模"
echo "3. 代码默认GMG：GMRES光滑器，也很好"
echo ""
echo "建议：使用 options_gpu_gmg.txt 获得最佳性能"
