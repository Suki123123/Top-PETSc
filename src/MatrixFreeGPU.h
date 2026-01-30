/**
 * MatrixFreeGPU.h - GPU实现的Matrix-Free接口
 */

#ifndef MATRIX_FREE_GPU_H
#define MATRIX_FREE_GPU_H

#include <petsc.h>

// 前向声明GPU资源结构
typedef struct GPUResources GPUResources;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 初始化GPU资源
 */
PetscErrorCode MatrixFreeGPU_Init(
    GPUResources** resources,
    const PetscScalar* KE,
    const PetscInt* necon,
    PetscInt nel,
    PetscInt n_local,
    PetscInt n_owned);

/**
 * 释放GPU资源
 */
PetscErrorCode MatrixFreeGPU_Destroy(GPUResources* res);

/**
 * GPU版本的MatMult
 */
PetscErrorCode MatrixFreeGPU_MatMult(
    GPUResources* res,
    PetscScalar* d_y,
    const PetscScalar* d_x_local,
    const PetscScalar* d_x_global,
    const PetscScalar* d_xPhys,
    const PetscScalar* d_N,
    PetscScalar Emin,
    PetscScalar Emax,
    PetscScalar penal);

/**
 * GPU版本的GetDiagonal
 */
PetscErrorCode MatrixFreeGPU_GetDiagonal(
    GPUResources* res,
    PetscScalar* d_diag,
    const PetscScalar* d_xPhys,
    const PetscScalar* d_N,
    PetscScalar Emin,
    PetscScalar Emax,
    PetscScalar penal);

/**
 * GPU版本的边界条件应用
 */
PetscErrorCode MatrixFreeGPU_ApplyBC(
    GPUResources* res,
    PetscScalar* d_y,
    const PetscScalar* d_x,
    const PetscScalar* d_N,
    PetscInt n);

/**
 * GPU版本的敏感度计算
 */
PetscErrorCode MatrixFreeGPU_ComputeSensitivity(
    GPUResources* res,
    PetscScalar* d_fx_partial,
    PetscScalar* d_dfdx,
    const PetscScalar* d_u_local,
    const PetscScalar* d_xPhys,
    PetscScalar Emin,
    PetscScalar Emax,
    PetscScalar penal);

/**
 * 获取单元数量
 */
PetscInt MatrixFreeGPU_GetNel(GPUResources* res);

// ============================================================================
// Filter相关GPU函数
// ============================================================================

/**
 * GPU版本的Heaviside投影滤波器
 */
PetscErrorCode FilterGPU_HeavisideFilter(
    PetscScalar* d_y,
    const PetscScalar* d_x,
    PetscInt n,
    PetscScalar beta,
    PetscScalar eta);

/**
 * GPU版本的Heaviside链式法则
 */
PetscErrorCode FilterGPU_ChainruleHeavisideFilter(
    PetscScalar* d_y,
    const PetscScalar* d_x,
    PetscInt n,
    PetscScalar beta,
    PetscScalar eta);

/**
 * GPU版本的MND计算
 */
PetscErrorCode FilterGPU_GetMND(
    PetscScalar* result,
    const PetscScalar* d_x,
    PetscInt n);

/**
 * GPU版本的元素级乘法
 */
PetscErrorCode FilterGPU_ElementwiseMult(
    PetscScalar* d_y,
    const PetscScalar* d_dx,
    PetscInt n);

/**
 * GPU版本的边界检查
 */
PetscErrorCode FilterGPU_BoundCheck(
    PetscScalar* d_x,
    PetscInt n);

// ============================================================================
// OC优化器相关GPU函数
// ============================================================================

/**
 * GPU版本的OC更新（给定lambda）
 */
PetscErrorCode OCGPU_Update(
    PetscScalar* d_x_new,
    const PetscScalar* d_x,
    const PetscScalar* d_dfdx,
    const PetscScalar* d_dgdx,
    PetscInt n,
    PetscScalar lambda,
    PetscScalar move,
    PetscScalar xmin,
    PetscScalar xmax);

/**
 * GPU版本的体积计算
 */
PetscErrorCode OCGPU_ComputeVolume(
    PetscScalar* result,
    const PetscScalar* d_x,
    PetscInt n);

/**
 * GPU版本的设计变化量计算
 */
PetscErrorCode OCGPU_ComputeChange(
    PetscScalar* result,
    const PetscScalar* d_x_new,
    const PetscScalar* d_x_old,
    PetscInt n);

/**
 * GPU版本的向量复制
 */
PetscErrorCode OCGPU_VecCopy(
    PetscScalar* d_dst,
    const PetscScalar* d_src,
    PetscInt n);

/**
 * GPU版本的OC优化器（带二分法求lambda）
 */
PetscErrorCode OCGPU_Optimize(
    PetscScalar* d_x,
    PetscScalar* d_x_old,
    PetscScalar* d_x_new,
    const PetscScalar* d_dfdx,
    const PetscScalar* d_dgdx,
    PetscInt n_local,
    PetscInt n_global,
    PetscScalar volfrac,
    PetscScalar move,
    PetscScalar* change);

/**
 * GPU版本的SetOuterMovelimit
 */
PetscErrorCode OCGPU_SetOuterMovelimit(
    PetscScalar* d_xmin,
    PetscScalar* d_xmax,
    const PetscScalar* d_x,
    PetscInt n,
    PetscScalar Xmin,
    PetscScalar Xmax,
    PetscScalar move);

/**
 * GPU版本的向量点除
 */
PetscErrorCode VecGPU_PointwiseDivide(
    PetscScalar* d_y,
    const PetscScalar* d_x,
    const PetscScalar* d_z,
    PetscInt n);

/**
 * GPU版本的向量点乘
 */
PetscErrorCode VecGPU_PointwiseMultiply(
    PetscScalar* d_y,
    const PetscScalar* d_x,
    const PetscScalar* d_z,
    PetscInt n);

#ifdef __cplusplus
}
#endif

#endif // MATRIX_FREE_GPU_H
