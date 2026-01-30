#include "LinearElasticity.h"
#include "MatrixFreeGPU.h"
#include <vector>
#include <petscvec.h>
#include <cuda_runtime.h>

/*
 Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013

 Disclaimer:
 The authors reserves all rights but does not guaranty that the code is
 free from errors. Furthermore, we shall not be liable in any event
 caused by the use of the program.
*/

// ============================================================================
// Matrix-Free矩阵销毁函数
// ============================================================================
static PetscErrorCode MatDestroy_MatrixFree(Mat A) {
    MatrixFreeContext* ctx;
    MatShellGetContext(A, &ctx);
    // 释放GPU资源
    if (ctx->gpu_res) {
        MatrixFreeGPU_Destroy(ctx->gpu_res);
    }
    delete ctx;
    return 0;
}

// ============================================================================
// Matrix-Free矩阵-向量乘积：y = A*x
// GPU版本：使用CUDA内核进行计算
// ============================================================================
static PetscErrorCode MatMult_MatrixFree(Mat A, Vec x, Vec y) {
    PetscErrorCode ierr;
    MatrixFreeContext* ctx;

    // 获取上下文
    ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);
    LinearElasticity* le = ctx->le;

    // 清零输出向量
    VecSet(y, 0.0);

    // 获取单元信息
    PetscInt nel, nen;
    const PetscInt* necon;
    ierr = le->DMDAGetElements_3D(ctx->da, &nel, &nen, &necon); CHKERRQ(ierr);

    // 创建局部向量 - 使用与DM关联的向量类型
    Vec xloc, yloc;
    DMGetLocalVector(ctx->da, &xloc);
    DMGetLocalVector(ctx->da, &yloc);

    // 全局到局部（包含halo交换）- PETSc自动处理GPU向量
    DMGlobalToLocalBegin(ctx->da, x, INSERT_VALUES, xloc);
    DMGlobalToLocalEnd(ctx->da, x, INSERT_VALUES, xloc);
    VecSet(yloc, 0.0);

    // 检查是否使用GPU
    PetscBool use_gpu = PETSC_FALSE;
    VecType vec_type;
    VecGetType(xloc, &vec_type);
    if (vec_type && ctx->gpu_res &&
        (strcmp(vec_type, VECCUDA) == 0 || strcmp(vec_type, VECMPICUDA) == 0 ||
         strcmp(vec_type, VECSEQCUDA) == 0)) {
        use_gpu = PETSC_TRUE;
    }

    if (use_gpu) {
        // ========== GPU路径 ==========
        // 获取GPU指针
        PetscScalar *d_yloc;
        const PetscScalar *d_xloc, *d_xPhys;

        ierr = VecCUDAGetArrayRead(xloc, &d_xloc); CHKERRQ(ierr);
        ierr = VecCUDAGetArray(yloc, &d_yloc); CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(ctx->xPhys, &d_xPhys); CHKERRQ(ierr);

        // 调用GPU内核（只执行单元循环）
        ierr = MatrixFreeGPU_MatMult(ctx->gpu_res, d_yloc, d_xloc, NULL, d_xPhys, NULL,
                                     ctx->Emin, ctx->Emax, ctx->penal); CHKERRQ(ierr);

        // 恢复GPU指针
        ierr = VecCUDARestoreArrayRead(xloc, &d_xloc); CHKERRQ(ierr);
        ierr = VecCUDARestoreArray(yloc, &d_yloc); CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(ctx->xPhys, &d_xPhys); CHKERRQ(ierr);

    } else {
        // ========== CPU路径（回退）==========
        PetscScalar* xp;
        VecGetArrayRead(ctx->xPhys, (const PetscScalar**)&xp);

        PetscScalar *xlocal, *ylocal;
        VecGetArrayRead(xloc, (const PetscScalar**)&xlocal);
        VecGetArray(yloc, &ylocal);

        // 单元循环
        PetscInt edof[24];
        PetscScalar xe[24], ye[24];

        for (PetscInt i = 0; i < nel; i++) {
            for (PetscInt j = 0; j < nen; j++) {
                for (PetscInt k = 0; k < 3; k++) {
                    edof[j * 3 + k] = 3 * necon[i * nen + j] + k;
                }
            }

            for (PetscInt j = 0; j < 24; j++) {
                xe[j] = xlocal[edof[j]];
            }

            PetscScalar dens = ctx->Emin + PetscPowScalar(xp[i], ctx->penal) * (ctx->Emax - ctx->Emin);

            for (PetscInt j = 0; j < 24; j++) {
                ye[j] = 0.0;
                for (PetscInt k = 0; k < 24; k++) {
                    ye[j] += le->KE[j * 24 + k] * xe[k];
                }
                ye[j] *= dens;
            }

            for (PetscInt j = 0; j < 24; j++) {
                ylocal[edof[j]] += ye[j];
            }
        }

        VecRestoreArrayRead(xloc, (const PetscScalar**)&xlocal);
        VecRestoreArray(yloc, &ylocal);
        VecRestoreArrayRead(ctx->xPhys, (const PetscScalar**)&xp);
    }

    // 局部到全局（包含MPI通信）
    DMLocalToGlobalBegin(ctx->da, yloc, ADD_VALUES, y);
    DMLocalToGlobalEnd(ctx->da, yloc, ADD_VALUES, y);

    // 应用Dirichlet边界条件
    Vec N_level = (ctx->level < 0) ? le->N : le->coarse_N[ctx->level];

    if (use_gpu) {
        // GPU路径：使用GPU内核应用边界条件
        PetscScalar* d_y;
        const PetscScalar *d_x, *d_N;
        PetscInt local_size;
        VecGetLocalSize(y, &local_size);

        ierr = VecCUDAGetArray(y, &d_y); CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(x, &d_x); CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(N_level, &d_N); CHKERRQ(ierr);

        ierr = MatrixFreeGPU_ApplyBC(ctx->gpu_res, d_y, d_x, d_N, local_size); CHKERRQ(ierr);

        ierr = VecCUDARestoreArray(y, &d_y); CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(x, &d_x); CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(N_level, &d_N); CHKERRQ(ierr);
    } else {
        // CPU路径
        PetscScalar* yarr;
        const PetscScalar* xarr;
        const PetscScalar* narr;
        VecGetArray(y, &yarr);
        VecGetArrayRead(x, &xarr);
        VecGetArrayRead(N_level, &narr);
        PetscInt local_size;
        VecGetLocalSize(y, &local_size);
        for (PetscInt i = 0; i < local_size; i++) {
            yarr[i] = narr[i] * yarr[i] + (1.0 - narr[i]) * xarr[i];
        }
        VecRestoreArray(y, &yarr);
        VecRestoreArrayRead(x, &xarr);
        VecRestoreArrayRead(N_level, &narr);
    }

    // 归还局部向量
    DMRestoreLocalVector(ctx->da, &xloc);
    DMRestoreLocalVector(ctx->da, &yloc);
    DMDARestoreElements(ctx->da, &nel, &nen, &necon);

    return 0;
}

// ============================================================================
// Matrix-Free获取对角线：用于Jacobi预条件器
// ============================================================================
static PetscErrorCode MatGetDiagonal_MatrixFree(Mat A, Vec d) {
    PetscErrorCode ierr;
    MatrixFreeContext* ctx;

    // 获取上下文
    ierr = MatShellGetContext(A, &ctx); CHKERRQ(ierr);
    LinearElasticity* le = ctx->le;

    // 清零对角线向量
    VecSet(d, 0.0);

    // 获取单元信息
    PetscInt nel, nen;
    const PetscInt* necon;
    ierr = le->DMDAGetElements_3D(ctx->da, &nel, &nen, &necon); CHKERRQ(ierr);

    // 创建局部向量
    Vec dloc;
    DMGetLocalVector(ctx->da, &dloc);
    VecSet(dloc, 0.0);

    // 检查是否使用GPU
    PetscBool use_gpu = PETSC_FALSE;
    VecType vec_type;
    VecGetType(dloc, &vec_type);
    if (vec_type && ctx->gpu_res &&
        (strcmp(vec_type, VECCUDA) == 0 || strcmp(vec_type, VECMPICUDA) == 0 ||
         strcmp(vec_type, VECSEQCUDA) == 0)) {
        use_gpu = PETSC_TRUE;
    }

    if (use_gpu) {
        // ========== GPU路径 ==========
        PetscScalar *d_dloc;
        const PetscScalar *d_xPhys;

        ierr = VecCUDAGetArray(dloc, &d_dloc); CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(ctx->xPhys, &d_xPhys); CHKERRQ(ierr);

        // 调用GPU内核
        ierr = MatrixFreeGPU_GetDiagonal(ctx->gpu_res, d_dloc, d_xPhys, NULL,
                                          ctx->Emin, ctx->Emax, ctx->penal); CHKERRQ(ierr);

        ierr = VecCUDARestoreArray(dloc, &d_dloc); CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(ctx->xPhys, &d_xPhys); CHKERRQ(ierr);

    } else {
        // ========== CPU路径 ==========
        PetscScalar* xp;
        VecGetArray(ctx->xPhys, &xp);

        PetscScalar* dlocal;
        VecGetArray(dloc, &dlocal);

        // 单元循环 - 累加对角线元素
        PetscInt edof[24];

        for (PetscInt i = 0; i < nel; i++) {
            // 获取单元DOF
            for (PetscInt j = 0; j < nen; j++) {
                for (PetscInt k = 0; k < 3; k++) {
                    edof[j * 3 + k] = 3 * necon[i * nen + j] + k;
                }
            }

            // 计算密度因子
            PetscScalar dens = ctx->Emin + PetscPowScalar(xp[i], ctx->penal) * (ctx->Emax - ctx->Emin);

            // 累加对角线元素 (KE的对角线元素 * 密度)
            for (PetscInt j = 0; j < 24; j++) {
                dlocal[edof[j]] += le->KE[j * 24 + j] * dens;
            }
        }

        VecRestoreArray(dloc, &dlocal);
        VecRestoreArray(ctx->xPhys, &xp);
    }

    // 局部到全局
    DMLocalToGlobalBegin(ctx->da, dloc, ADD_VALUES, d);
    DMLocalToGlobalEnd(ctx->da, dloc, ADD_VALUES, d);

    // 应用Dirichlet边界条件
    Vec N_level;
    if (ctx->level < 0) {
        N_level = le->N;
    } else {
        N_level = le->coarse_N[ctx->level];
    }

    // 对于固定DOF，对角线设为1.0: d = d * N + (1 - N)
    if (use_gpu) {
        PetscScalar *d_d;
        const PetscScalar *d_N;
        PetscInt local_size;
        VecGetLocalSize(d, &local_size);

        ierr = VecCUDAGetArray(d, &d_d); CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(N_level, &d_N); CHKERRQ(ierr);

        ierr = MatrixFreeGPU_ApplyBC(ctx->gpu_res, d_d, d_d, d_N, local_size); CHKERRQ(ierr);

        ierr = VecCUDARestoreArray(d, &d_d); CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(N_level, &d_N); CHKERRQ(ierr);
    } else {
        PetscScalar* darr;
        const PetscScalar* narr;
        VecGetArray(d, &darr);
        VecGetArrayRead(N_level, &narr);
        PetscInt local_size;
        VecGetLocalSize(d, &local_size);
        for (PetscInt i = 0; i < local_size; i++) {
            darr[i] = darr[i] * narr[i] + (1.0 - narr[i]);
        }
        VecRestoreArray(d, &darr);
        VecRestoreArrayRead(N_level, &narr);
    }

    // 清理
    DMRestoreLocalVector(ctx->da, &dloc);
    DMDARestoreElements(ctx->da, &nel, &nen, &necon);

    return 0;
}

LinearElasticity::LinearElasticity(DM da_nodes) {
    // Set pointers to null
    K   = NULL;
    U   = NULL;
    RHS = NULL;
    N   = NULL;
    ksp = NULL;
    da_nodal;

    // 多层网格指针初始化
    coarse_K = NULL;
    coarse_da = NULL;
    coarse_N = NULL;
    density_levels = NULL;
    interpolation = NULL;
    use_geometric_mg = PETSC_FALSE;

    // Matrix-Free相关初始化 - 默认启用
    use_matrix_free = PETSC_TRUE;
    mf_ctx = NULL;
    mf_ctx_levels = NULL;

    // Parameters - to be changed on read of variables
    nu    = 0.3;
    nlvls = 4;
    PetscBool flg;
    PetscOptionsGetInt(NULL, NULL, "-nlvls", &nlvls, &flg);
    PetscOptionsGetReal(NULL, NULL, "-nu", &nu, &flg);

    PetscPrintf(PETSC_COMM_WORLD, "# Matrix-Free mode (default)\n");

    // Setup sitffness matrix, load vector and bcs (Dirichlet) for the design
    // problem
    SetUpLoadAndBC(da_nodes);
}

LinearElasticity::~LinearElasticity() {
    // Deallocate
    VecDestroy(&(U));
    VecDestroy(&(RHS));
    VecDestroy(&(N));
    MatDestroy(&(K));
    KSPDestroy(&(ksp));

    if (da_nodal != NULL) {
        DMDestroy(&(da_nodal));
    }

    // 释放多层网格资源
    if (coarse_K != NULL) {
        for (PetscInt k = 0; k < nlvls - 1; k++) {
            if (coarse_K[k] != NULL) MatDestroy(&coarse_K[k]);
        }
        delete[] coarse_K;
    }
    if (coarse_da != NULL) {
        for (PetscInt k = 0; k < nlvls - 1; k++) {
            if (coarse_da[k] != NULL) DMDestroy(&coarse_da[k]);
        }
        delete[] coarse_da;
    }
    if (coarse_N != NULL) {
        for (PetscInt k = 0; k < nlvls - 1; k++) {
            if (coarse_N[k] != NULL) VecDestroy(&coarse_N[k]);
        }
        delete[] coarse_N;
    }
    if (density_levels != NULL) {
        for (PetscInt k = 0; k < nlvls; k++) {
            if (density_levels[k] != NULL) VecDestroy(&density_levels[k]);
        }
        delete[] density_levels;
    }
    if (interpolation != NULL) {
        for (PetscInt k = 0; k < nlvls - 1; k++) {
            if (interpolation[k] != NULL) MatDestroy(&interpolation[k]);
        }
        delete[] interpolation;
    }

    // 释放Matrix-Free资源
    // 注意：mf_ctx和mf_ctx_levels中的上下文会在MatDestroy时自动释放
    if (mf_ctx_levels != NULL) {
        delete[] mf_ctx_levels;
    }
}

PetscErrorCode LinearElasticity::SetUpLoadAndBC(DM da_nodes) {

    PetscErrorCode ierr;
    // Extract information from input DM and create one for the linear elasticity
    // number of nodal dofs: (u,v,w)
    PetscInt numnodaldof = 3;

    // Stencil width: each node connects to a box around it - linear elements
    PetscInt stencilwidth = 1;

    PetscScalar     dx, dy, dz;
    DMBoundaryType  bx, by, bz;
    DMDAStencilType stype;
    {
        // Extract information from the nodal mesh
        PetscInt M, N, P, md, nd, pd;
        DMDAGetInfo(da_nodes, NULL, &M, &N, &P, &md, &nd, &pd, NULL, NULL, &bx, &by, &bz, &stype);

        // Find the element size
        Vec lcoor;
        DMGetCoordinatesLocal(da_nodes, &lcoor);
        PetscScalar* lcoorp;
        VecGetArray(lcoor, &lcoorp);

        PetscInt        nel, nen;
        const PetscInt* necon;
        DMDAGetElements_3D(da_nodes, &nel, &nen, &necon);

        // Use the first element to compute the dx, dy, dz
        dx = lcoorp[3 * necon[0 * nen + 1] + 0] - lcoorp[3 * necon[0 * nen + 0] + 0];
        dy = lcoorp[3 * necon[0 * nen + 2] + 1] - lcoorp[3 * necon[0 * nen + 1] + 1];
        dz = lcoorp[3 * necon[0 * nen + 4] + 2] - lcoorp[3 * necon[0 * nen + 0] + 2];
        VecRestoreArray(lcoor, &lcoorp);

        nn[0] = M;
        nn[1] = N;
        nn[2] = P;

        ne[0] = nn[0] - 1;
        ne[1] = nn[1] - 1;
        ne[2] = nn[2] - 1;

        xc[0] = 0.0;
        xc[1] = ne[0] * dx;
        xc[2] = 0.0;
        xc[3] = ne[1] * dy;
        xc[4] = 0.0;
        xc[5] = ne[2] * dz;
    }

    // Create the nodal mesh
    DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, stype, nn[0], nn[1], nn[2], PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                 numnodaldof, stencilwidth, 0, 0, 0, &(da_nodal));
    // Initialize
    DMSetFromOptions(da_nodal);
    DMSetUp(da_nodal);

    // Set the coordinates
    DMDASetUniformCoordinates(da_nodal, xc[0], xc[1], xc[2], xc[3], xc[4], xc[5]);
    // Set the element type to Q1: Otherwise calls to GetElements will change to
    // P1 ! STILL DOESN*T WORK !!!!
    DMDASetElementType(da_nodal, DMDA_ELEMENT_Q1);

    // Allocate matrix and the RHS and Solution vector and Dirichlet vector
    ierr = DMCreateMatrix(da_nodal, &(K));
    CHKERRQ(ierr);
    // Allow runtime override of matrix type (e.g., -mat_type aijcusparse for GPU)
    ierr = MatSetFromOptions(K);
    CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da_nodal, &(U));
    CHKERRQ(ierr);
    VecDuplicate(U, &(RHS));
    VecDuplicate(U, &(N));

    // Set the local stiffness matrix
    PetscScalar X[8] = {0.0, dx, dx, 0.0, 0.0, dx, dx, 0.0};
    PetscScalar Y[8] = {0.0, 0.0, dy, dy, 0.0, 0.0, dy, dy};
    PetscScalar Z[8] = {0.0, 0.0, 0.0, 0.0, dz, dz, dz, dz};

    // Compute the element stiffnes matrix - constant due to structured grid
    Hex8Isoparametric(X, Y, Z, nu, false, KE);

    // Set the RHS and Dirichlet vector
    VecSet(N, 1.0);
    VecSet(RHS, 0.0);

    // Global coordinates and a pointer
    Vec          lcoor; // borrowed ref - do not destroy!
    PetscScalar* lcoorp;

    // Get local coordinates in local node numbering including ghosts
    ierr = DMGetCoordinatesLocal(da_nodal, &lcoor);
    CHKERRQ(ierr);
    VecGetArray(lcoor, &lcoorp);

    // Get local dof number
    PetscInt nn;
    VecGetSize(lcoor, &nn);

    // Compute epsilon parameter for finding points in space:
    PetscScalar epsi = PetscMin(dx * 0.05, PetscMin(dy * 0.05, dz * 0.05));

    // Set the values:
    // Cantilever beam: Left face fixed, load on bottom edge of right face
    // N = the wall at x=xmin is fully clamped
    // RHS(z) = -1.0 at x=xmax, z=zmin (bottom edge)
    PetscScalar LoadIntensity = -1.0;
    for (PetscInt i = 0; i < nn; i++) {
        // Make a wall with all dofs clamped
        if (i % 3 == 0 && PetscAbsScalar(lcoorp[i] - xc[0]) < epsi) {
            VecSetValueLocal(N, i, 0.0, INSERT_VALUES);
            VecSetValueLocal(N, ++i, 0.0, INSERT_VALUES);
            VecSetValueLocal(N, ++i, 0.0, INSERT_VALUES);
        }
        // Line load
        if (i % 3 == 0 && PetscAbsScalar(lcoorp[i] - xc[1]) < epsi && PetscAbsScalar(lcoorp[i + 2] - xc[4]) < epsi) {
            VecSetValueLocal(RHS, i + 2, LoadIntensity, INSERT_VALUES);
        }
        // Adjust the corners
        if (i % 3 == 0 && PetscAbsScalar(lcoorp[i] - xc[1]) < epsi && PetscAbsScalar(lcoorp[i + 1] - xc[2]) < epsi &&
            PetscAbsScalar(lcoorp[i + 2] - xc[4]) < epsi) {
            VecSetValueLocal(RHS, i + 2, LoadIntensity / 2.0, INSERT_VALUES);
        }
        if (i % 3 == 0 && PetscAbsScalar(lcoorp[i] - xc[1]) < epsi && PetscAbsScalar(lcoorp[i + 1] - xc[3]) < epsi &&
            PetscAbsScalar(lcoorp[i + 2] - xc[4]) < epsi) {
            VecSetValueLocal(RHS, i + 2, LoadIntensity / 2.0, INSERT_VALUES);
        }
    }

    VecAssemblyBegin(N);
    VecAssemblyBegin(RHS);
    VecAssemblyEnd(N);
    VecAssemblyEnd(RHS);
    VecRestoreArray(lcoor, &lcoorp);

    return ierr;
}

PetscErrorCode LinearElasticity::SolveState(Vec xPhys, PetscScalar Emin, PetscScalar Emax, PetscScalar penal) {

    PetscErrorCode ierr;

    double t1, t2;
    t1 = MPI_Wtime();

    // Matrix-Free模式：不组装矩阵，只更新上下文
    // 更新材料参数
    current_Emin = Emin;
    current_Emax = Emax;
    current_penal = penal;

    // Setup the solver (first time only)
    if (ksp == NULL) {
        // 设置求解器（会创建Shell矩阵）
        ierr = SetUpSolver();
        CHKERRQ(ierr);

        // 初始化密度向量（在SetUpSolver之后，因为use_geometric_mg在SetUpSolver中设置）
        if (use_geometric_mg) {
            ierr = RestrictDensity(xPhys);
            CHKERRQ(ierr);
        }

        // 更新Matrix-Free上下文（使用实际的xPhys）
        MatrixFreeContext* ctx;
        MatShellGetContext(K, &ctx);
        ctx->xPhys = xPhys;
        ctx->Emin = Emin;
        ctx->Emax = Emax;
        ctx->penal = penal;

        // 更新粗网格上下文
        if (use_geometric_mg && mf_ctx_levels != NULL) {
            for (PetscInt k = 0; k < nlvls - 1; k++) {
                if (mf_ctx_levels[k] != NULL) {
                    mf_ctx_levels[k]->xPhys = density_levels[k];
                    mf_ctx_levels[k]->Emin = Emin;
                    mf_ctx_levels[k]->Emax = Emax;
                    mf_ctx_levels[k]->penal = penal;
                }
            }
        }
    } else {
        // 更新密度限制
        if (use_geometric_mg) {
            ierr = RestrictDensity(xPhys);
            CHKERRQ(ierr);
        }

        // 更新Matrix-Free上下文
        MatrixFreeContext* ctx;
        MatShellGetContext(K, &ctx);
        ctx->xPhys = xPhys;
        ctx->Emin = Emin;
        ctx->Emax = Emax;
        ctx->penal = penal;

        // 更新粗网格上下文
        if (use_geometric_mg && mf_ctx_levels != NULL) {
            for (PetscInt k = 0; k < nlvls - 1; k++) {
                if (mf_ctx_levels[k] != NULL) {
                    mf_ctx_levels[k]->xPhys = density_levels[k];
                    mf_ctx_levels[k]->Emin = Emin;
                    mf_ctx_levels[k]->Emax = Emax;
                    mf_ctx_levels[k]->penal = penal;
                }
            }
        }
    }

    // 重新设置算子（触发KSP更新）
    ierr = KSPSetOperators(ksp, K, K);
    CHKERRQ(ierr);

    // Solve
    ierr = KSPSolve(ksp, RHS, U);
    CHKERRQ(ierr);

    // Get iteration number and residual from KSP
    PetscInt    niter;
    PetscScalar rnorm;
    KSPGetIterationNumber(ksp, &niter);
    KSPGetResidualNorm(ksp, &rnorm);
    PetscReal RHSnorm;
    ierr = VecNorm(RHS, NORM_2, &RHSnorm);
    CHKERRQ(ierr);
    rnorm = rnorm / RHSnorm;

    t2 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "State solver:  iter: %i, rerr.: %e, time: %f\n", niter, rnorm, t2 - t1);

    return ierr;
}

PetscErrorCode LinearElasticity::ComputeObjectiveConstraints(PetscScalar* fx, PetscScalar* gx, Vec xPhys,
                                                             PetscScalar Emin, PetscScalar Emax, PetscScalar penal,
                                                             PetscScalar volfrac) {

    // Error code
    PetscErrorCode ierr;

    // Solve state eqs
    ierr = SolveState(xPhys, Emin, Emax, penal);
    CHKERRQ(ierr);

    // Get the FE mesh structure (from the nodal mesh)
    PetscInt        nel, nen;
    const PetscInt* necon;
    ierr = DMDAGetElements_3D(da_nodal, &nel, &nen, &necon);
    CHKERRQ(ierr);

    // Get pointer to the densities
    PetscScalar* xp;
    VecGetArray(xPhys, &xp);

    // Get Solution
    Vec Uloc;
    DMCreateLocalVector(da_nodal, &Uloc);
    DMGlobalToLocalBegin(da_nodal, U, INSERT_VALUES, Uloc);
    DMGlobalToLocalEnd(da_nodal, U, INSERT_VALUES, Uloc);

    // get pointer to local vector
    PetscScalar* up;
    VecGetArray(Uloc, &up);

    // Edof array
    PetscInt edof[24];

    fx[0] = 0.0;
    // Loop over elements
    for (PetscInt i = 0; i < nel; i++) {
        // loop over element nodes
        for (PetscInt j = 0; j < nen; j++) {
            // Get local dofs
            for (PetscInt k = 0; k < 3; k++) {
                edof[j * 3 + k] = 3 * necon[i * nen + j] + k;
            }
        }
        // Use SIMP for stiffness interpolation
        PetscScalar uKu = 0.0;
        for (PetscInt k = 0; k < 24; k++) {
            for (PetscInt h = 0; h < 24; h++) {
                uKu += up[edof[k]] * KE[k * 24 + h] * up[edof[h]];
            }
        }
        // Add to objective
        fx[0] += (Emin + PetscPowScalar(xp[i], penal) * (Emax - Emin)) * uKu;
    }

    // Allreduce fx[0]
    PetscScalar tmp = fx[0];
    fx[0]           = 0.0;
    MPI_Allreduce(&tmp, &(fx[0]), 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

    // Compute volume constraint gx[0]
    PetscInt neltot;
    VecGetSize(xPhys, &neltot);
    gx[0] = 0;
    VecSum(xPhys, &(gx[0]));
    gx[0] = gx[0] / (((PetscScalar)neltot)) - volfrac;

    VecRestoreArray(xPhys, &xp);
    VecRestoreArray(Uloc, &up);
    VecDestroy(&Uloc);

    return (ierr);
}

PetscErrorCode LinearElasticity::ComputeSensitivities(Vec dfdx, Vec dgdx, Vec xPhys, PetscScalar Emin, PetscScalar Emax,
                                                      PetscScalar penal, PetscScalar volfrac) {

    PetscErrorCode ierr;

    // Get the FE mesh structure (from the nodal mesh)
    PetscInt        nel, nen;
    const PetscInt* necon;
    ierr = DMDAGetElements_3D(da_nodal, &nel, &nen, &necon);
    CHKERRQ(ierr);

    // Get pointer to the densities
    PetscScalar* xp;
    VecGetArray(xPhys, &xp);

    // Get Solution
    Vec Uloc;
    DMCreateLocalVector(da_nodal, &Uloc);
    DMGlobalToLocalBegin(da_nodal, U, INSERT_VALUES, Uloc);
    DMGlobalToLocalEnd(da_nodal, U, INSERT_VALUES, Uloc);

    // get pointer to local vector
    PetscScalar* up;
    VecGetArray(Uloc, &up);

    // Get dfdx
    PetscScalar* df;
    VecGetArray(dfdx, &df);

    // Edof array
    PetscInt edof[24];

    // Loop over elements
    for (PetscInt i = 0; i < nel; i++) {
        // loop over element nodes
        for (PetscInt j = 0; j < nen; j++) {
            // Get local dofs
            for (PetscInt k = 0; k < 3; k++) {
                edof[j * 3 + k] = 3 * necon[i * nen + j] + k;
            }
        }
        // Use SIMP for stiffness interpolation
        PetscScalar uKu = 0.0;
        for (PetscInt k = 0; k < 24; k++) {
            for (PetscInt h = 0; h < 24; h++) {
                uKu += up[edof[k]] * KE[k * 24 + h] * up[edof[h]];
            }
        }
        // Set the Senstivity
        df[i] = -1.0 * penal * PetscPowScalar(xp[i], penal - 1) * (Emax - Emin) * uKu;
    }
    // Compute volume constraint gx[0]
    PetscInt neltot;
    VecGetSize(xPhys, &neltot);
    VecSet(dgdx, 1.0 / (((PetscScalar)neltot)));

    VecRestoreArray(xPhys, &xp);
    VecRestoreArray(Uloc, &up);
    VecRestoreArray(dfdx, &df);
    VecDestroy(&Uloc);

    return (ierr);
}

PetscErrorCode LinearElasticity::ComputeObjectiveConstraintsSensitivities(PetscScalar* fx, PetscScalar* gx, Vec dfdx,
                                                                          Vec dgdx, Vec xPhys, PetscScalar Emin,
                                                                          PetscScalar Emax, PetscScalar penal,
                                                                          PetscScalar volfrac) {
    // Errorcode
    PetscErrorCode ierr;

    // Solve state eqs
    ierr = SolveState(xPhys, Emin, Emax, penal);
    CHKERRQ(ierr);

    // Get the FE mesh structure (from the nodal mesh)
    PetscInt        nel, nen;
    const PetscInt* necon;
    ierr = DMDAGetElements_3D(da_nodal, &nel, &nen, &necon);
    CHKERRQ(ierr);

    // Get Solution - create local vector with ghost nodes
    Vec Uloc;
    DMCreateLocalVector(da_nodal, &Uloc);
    DMGlobalToLocalBegin(da_nodal, U, INSERT_VALUES, Uloc);
    DMGlobalToLocalEnd(da_nodal, U, INSERT_VALUES, Uloc);

    // Check if GPU is available
    PetscBool use_gpu = PETSC_FALSE;
    VecType vec_type;
    VecGetType(Uloc, &vec_type);

    // Also check dfdx and xPhys vector types
    VecType dfdx_type, xPhys_type;
    VecGetType(dfdx, &dfdx_type);
    VecGetType(xPhys, &xPhys_type);

    // 检查是否可以使用GPU
    // 只有当所有向量都是CUDA向量时才使用GPU
    PetscBool uloc_cuda = PETSC_FALSE;
    PetscBool dfdx_cuda = PETSC_FALSE;
    PetscBool xPhys_cuda = PETSC_FALSE;

    if (vec_type && (strcmp(vec_type, VECCUDA) == 0 || strcmp(vec_type, VECMPICUDA) == 0 || strcmp(vec_type, VECSEQCUDA) == 0)) {
        uloc_cuda = PETSC_TRUE;
    }
    if (dfdx_type && (strcmp(dfdx_type, VECCUDA) == 0 || strcmp(dfdx_type, VECMPICUDA) == 0 || strcmp(dfdx_type, VECSEQCUDA) == 0)) {
        dfdx_cuda = PETSC_TRUE;
    }
    if (xPhys_type && (strcmp(xPhys_type, VECCUDA) == 0 || strcmp(xPhys_type, VECMPICUDA) == 0 || strcmp(xPhys_type, VECSEQCUDA) == 0)) {
        xPhys_cuda = PETSC_TRUE;
    }

    if (uloc_cuda && dfdx_cuda && xPhys_cuda && mf_ctx && mf_ctx->gpu_res) {
        use_gpu = PETSC_TRUE;
    }

    fx[0] = 0.0;

    if (use_gpu) {
        // ========== GPU路径 ==========
        const PetscScalar *d_uloc, *d_xPhys;
        PetscScalar *d_dfdx;

        // 获取GPU指针
        ierr = VecCUDAGetArrayRead(Uloc, &d_uloc); CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(xPhys, &d_xPhys); CHKERRQ(ierr);
        ierr = VecCUDAGetArray(dfdx, &d_dfdx); CHKERRQ(ierr);

        // 分配临时GPU内存用于部分目标函数值
        PetscScalar* d_fx_partial;
        cudaMalloc(&d_fx_partial, nel * sizeof(PetscScalar));

        // 调用GPU内核
        ierr = MatrixFreeGPU_ComputeSensitivity(mf_ctx->gpu_res, d_fx_partial, d_dfdx,
                                                 d_uloc, d_xPhys, Emin, Emax, penal);
        CHKERRQ(ierr);

        // 将部分目标函数值复制回CPU并求和
        PetscScalar* fx_partial = new PetscScalar[nel];
        cudaMemcpy(fx_partial, d_fx_partial, nel * sizeof(PetscScalar), cudaMemcpyDeviceToHost);

        for (PetscInt i = 0; i < nel; i++) {
            fx[0] += fx_partial[i];
        }

        // 清理
        delete[] fx_partial;
        cudaFree(d_fx_partial);

        // 恢复GPU指针
        ierr = VecCUDARestoreArrayRead(Uloc, &d_uloc); CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(xPhys, &d_xPhys); CHKERRQ(ierr);
        ierr = VecCUDARestoreArray(dfdx, &d_dfdx); CHKERRQ(ierr);

    } else {
        // ========== CPU路径（回退）==========
        // Get pointer to the densities
        PetscScalar* xp;
        VecGetArray(xPhys, &xp);

        // get pointer to local vector
        PetscScalar* up;
        VecGetArray(Uloc, &up);

        // Get dfdx
        PetscScalar* df;
        VecGetArray(dfdx, &df);

        // Edof array
        PetscInt edof[24];

        // Loop over elements
        for (PetscInt i = 0; i < nel; i++) {
            // loop over element nodes
            for (PetscInt j = 0; j < nen; j++) {
                // Get local dofs
                for (PetscInt k = 0; k < 3; k++) {
                    edof[j * 3 + k] = 3 * necon[i * nen + j] + k;
                }
            }
            // Use SIMP for stiffness interpolation
            PetscScalar uKu = 0.0;
            for (PetscInt k = 0; k < 24; k++) {
                for (PetscInt h = 0; h < 24; h++) {
                    uKu += up[edof[k]] * KE[k * 24 + h] * up[edof[h]];
                }
            }
            // Add to objective
            fx[0] += (Emin + PetscPowScalar(xp[i], penal) * (Emax - Emin)) * uKu;
            // Set the Senstivity
            df[i] = -1.0 * penal * PetscPowScalar(xp[i], penal - 1) * (Emax - Emin) * uKu;
        }

        VecRestoreArray(xPhys, &xp);
        VecRestoreArray(Uloc, &up);
        VecRestoreArray(dfdx, &df);
    }

    // Allreduce fx[0]
    PetscScalar tmp = fx[0];
    fx[0]           = 0.0;
    MPI_Allreduce(&tmp, &(fx[0]), 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

    // Compute volume constraint gx[0]
    PetscInt neltot;
    VecGetSize(xPhys, &neltot);
    gx[0] = 0;
    VecSum(xPhys, &(gx[0]));
    gx[0] = gx[0] / (((PetscScalar)neltot)) - volfrac;
    VecSet(dgdx, 1.0 / (((PetscScalar)neltot)));

    VecDestroy(&Uloc);
    DMDARestoreElements(da_nodal, &nel, &nen, &necon);

    return (ierr);
}

PetscErrorCode LinearElasticity::WriteRestartFiles() {

    PetscErrorCode ierr = 0;

    // Only dump data if correct allocater has been used
    if (!restart) {
        return -1;
    }

    // Choose previous set of restart files
    if (flip) {
        flip = PETSC_FALSE;
    } else {
        flip = PETSC_TRUE;
    }

    // Open viewers for writing
    PetscViewer view; // vectors
    if (!flip) {
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename00.c_str(), FILE_MODE_WRITE, &view);
    } else if (flip) {
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename01.c_str(), FILE_MODE_WRITE, &view);
    }

    // Write vectors
    VecView(U, view);

    // Clean up
    PetscViewerDestroy(&view);

    return ierr;
}

//##################################################################
//##################################################################
//##################################################################
// ######################## PRIVATE ################################
//##################################################################
//##################################################################

PetscErrorCode LinearElasticity::SetUpSolver() {

    PetscErrorCode ierr;

    // CHECK FOR RESTART POINT
    restart = PETSC_TRUE;
    flip    = PETSC_TRUE;
    PetscBool flg, onlyDesign;
    onlyDesign = PETSC_FALSE;
    char filenameChar[PETSC_MAX_PATH_LEN];
    PetscOptionsGetBool(NULL, NULL, "-restart", &restart, &flg);
    PetscOptionsGetBool(NULL, NULL, "-onlyLoadDesign", &onlyDesign,
                        &flg); // DONT READ DESIGN IF THIS IS TRUE

    // READ THE RESTART FILE INTO THE SOLUTION VECTOR(S)
    if (restart) {
        // THE FILES FOR WRITING RESTARTS
        std::string filenameWorkdir = "./";
        PetscOptionsGetString(NULL, NULL, "-workdir", filenameChar, sizeof(filenameChar), &flg);
        if (flg) {
            filenameWorkdir = "";
            filenameWorkdir.append(filenameChar);
        }
        filename00 = filenameWorkdir;
        filename01 = filenameWorkdir;
        filename00.append("/RestartSol00.dat");
        filename01.append("/RestartSol01.dat");

        // CHECK FOR SOLUTION AND READ TO STATE VECTOR(s)
        if (!onlyDesign) {
            // Where to read the restart point from
            std::string restartFileVec = ""; // NO RESTART FILE !!!!!
            // GET FILENAME
            PetscOptionsGetString(NULL, NULL, "-restartFileVecSol", filenameChar, sizeof(filenameChar), &flg);
            if (flg) {
                restartFileVec.append(filenameChar);
            }

            // PRINT TO SCREEN
            PetscPrintf(PETSC_COMM_WORLD,
                        "# Restarting with solution (State Vector) from "
                        "(-restartFileVecSol): %s \n",
                        restartFileVec.c_str());

            // Check if files exist:
            PetscBool vecFile = fexists(restartFileVec);
            if (!vecFile) {
                PetscPrintf(PETSC_COMM_WORLD, "File: %s NOT FOUND \n", restartFileVec.c_str());
            }

            // READ
            if (vecFile) {
                PetscViewer view;
                // Open the data files
                ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, restartFileVec.c_str(), FILE_MODE_READ, &view);

                VecLoad(U, view);

                PetscViewerDestroy(&view);
            }
        }
    }

    PC pc;

    // The fine grid Krylov method
    KSPCreate(PETSC_COMM_WORLD, &(ksp));

    // SET THE DEFAULT SOLVER PARAMETERS
    // The fine grid solver settings
    PetscScalar rtol         = 1.0e-5;
    PetscScalar atol         = 1.0e-50;
    PetscScalar dtol         = 1.0e5;
    PetscInt    restart      = 100;
    PetscInt    maxitsGlobal = 200;

    // Coarsegrid solver
    PetscScalar coarse_rtol    = 1.0e-8;
    PetscScalar coarse_atol    = 1.0e-50;
    PetscScalar coarse_dtol    = 1e5;
    PetscInt    coarse_maxits  = 30;
    PetscInt    coarse_restart = 30;

    // Number of smoothening iterations per up/down smooth_sweeps
    PetscInt smooth_sweeps = 4;

    // Set up the solver
    ierr = KSPSetType(ksp, KSPFGMRES); // KSPCG, KSPGMRES
    CHKERRQ(ierr);

    ierr = KSPGMRESSetRestart(ksp, restart);
    CHKERRQ(ierr);

    ierr = KSPSetTolerances(ksp, rtol, atol, dtol, maxitsGlobal);
    CHKERRQ(ierr);

    ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    CHKERRQ(ierr);

    // Matrix-Free模式：创建Shell矩阵
    PetscPrintf(PETSC_COMM_WORLD, "# Creating Matrix-Free Shell matrix for finest level\n");
    MatDestroy(&K);

    // 创建临时密度向量用于初始化
    Vec xPhys_temp;
    DMCreateGlobalVector(da_nodal, &xPhys_temp);
    VecSet(xPhys_temp, 0.5);  // 使用默认密度值

    ierr = CreateShellMatrix(da_nodal, xPhys_temp, &K, current_Emin, current_Emax, current_penal, -1);
    CHKERRQ(ierr);

    // 保存最细层上下文（用于敏感度计算等）
    MatShellGetContext(K, &mf_ctx);

    // 保存临时密度向量的引用（稍后会被实际的xPhys替换）
    // 注意：这个向量会在SolveState中被替换

    ierr = KSPSetOperators(ksp, K, K);
    CHKERRQ(ierr);

    // The preconditinoer
    KSPGetPC(ksp, &pc);
    // Make PCMG the default solver
    PCSetType(pc, PCMG);

    // Set solver from options
    KSPSetFromOptions(ksp);

    // Get the prec again - check if it has changed
    KSPGetPC(ksp, &pc);

    // Flag for pcmg pc
    PetscBool pcmg_flag = PETSC_TRUE;
    PetscObjectTypeCompare((PetscObject)pc, PCMG, &pcmg_flag);

    // Only if PCMG is used
    if (pcmg_flag) {

        // DMs for grid hierachy
        DM *da_list, *daclist;

        PetscMalloc(sizeof(DM) * nlvls, &da_list);
        for (PetscInt k = 0; k < nlvls; k++)
            da_list[k] = NULL;
        PetscMalloc(sizeof(DM) * nlvls, &daclist);
        for (PetscInt k = 0; k < nlvls; k++)
            daclist[k] = NULL;

        // Set 0 to the finest level
        daclist[0] = da_nodal;

        // Coordinates
        PetscReal xmin = xc[0], xmax = xc[1], ymin = xc[2], ymax = xc[3], zmin = xc[4], zmax = xc[5];

        // Set up the coarse meshes
        DMCoarsenHierarchy(da_nodal, nlvls - 1, &daclist[1]);
        for (PetscInt k = 0; k < nlvls; k++) {
            // NOTE: finest grid is nlevels - 1: PCMG MUST USE THIS ORDER ???
            da_list[k] = daclist[nlvls - 1 - k];
            // THIS SHOULD NOT BE NECESSARY
            DMDASetUniformCoordinates(da_list[k], xmin, xmax, ymin, ymax, zmin, zmax);
        }

        // the PCMG specific options
        PCMGSetLevels(pc, nlvls, NULL);
        PCMGSetType(pc, PC_MG_MULTIPLICATIVE); // Default
        ierr = PCMGSetCycleType(pc, PC_MG_CYCLE_V);
        CHKERRQ(ierr);
        
        // 使用几何重离散化（通过回调函数）
        PCMGSetGalerkin(pc, PC_MG_GALERKIN_NONE);
        use_geometric_mg = PETSC_TRUE;
        
        // 分配粗网格资源
        coarse_K = new Mat[nlvls - 1];
        coarse_da = new DM[nlvls - 1];
        coarse_N = new Vec[nlvls - 1];
        density_levels = new Vec[nlvls];  // nlvls个密度向量（包括最细层）
        interpolation = new Mat[nlvls - 1];
        
        for (PetscInt k = 0; k < nlvls - 1; k++) {
            coarse_K[k] = NULL;
            coarse_da[k] = NULL;
            coarse_N[k] = NULL;
            interpolation[k] = NULL;
        }
        for (PetscInt k = 0; k < nlvls; k++) {
            density_levels[k] = NULL;
        }

        // 设置插值算子并保存粗网格DM
        for (PetscInt k = 1; k < nlvls; k++) {
            Mat R;
            DMCreateInterpolation(da_list[k - 1], da_list[k], &R, NULL);
            PCMGSetInterpolation(pc, k, R);

            // 保存插值算子（用于密度限制）
            interpolation[k - 1] = R;
            PetscObjectReference((PetscObject)R);
            MatDestroy(&R);

            // 保存粗网格DM（需要增加引用计数）
            coarse_da[k - 1] = da_list[k - 1];
            PetscObjectReference((PetscObject)coarse_da[k - 1]);

            // 为粗网格创建密度向量（初始化为默认值）
            DMCreateGlobalVector(coarse_da[k - 1], &density_levels[k - 1]);
            VecSet(density_levels[k - 1], 0.5);  // 使用默认密度值

            // 为粗网格创建Shell矩阵（使用密度向量）
            ierr = CreateShellMatrix(coarse_da[k - 1], density_levels[k - 1], &coarse_K[k - 1],
                                    current_Emin, current_Emax, current_penal, k - 1);
            CHKERRQ(ierr);

            // 为粗网格创建Dirichlet向量并设置边界条件
            DMCreateGlobalVector(coarse_da[k - 1], &coarse_N[k - 1]);
            VecSet(coarse_N[k - 1], 1.0);

            // 设置粗网格边界条件（与细网格相同的逻辑）
            {
                Vec lcoor;
                PetscScalar* lcoorp;
                ierr = DMGetCoordinatesLocal(coarse_da[k - 1], &lcoor);
                CHKERRQ(ierr);
                VecGetArray(lcoor, &lcoorp);

                PetscInt nn_local;
                VecGetSize(lcoor, &nn_local);

                PetscScalar epsi = PetscMin(xc[1] - xc[0], PetscMin(xc[3] - xc[2], xc[5] - xc[4])) * 0.05;

                for (PetscInt i = 0; i < nn_local; i++) {
                    // 在x=xmin处固定所有自由度
                    if (i % 3 == 0 && PetscAbsScalar(lcoorp[i] - xc[0]) < epsi) {
                        VecSetValueLocal(coarse_N[k - 1], i, 0.0, INSERT_VALUES);
                        VecSetValueLocal(coarse_N[k - 1], i + 1, 0.0, INSERT_VALUES);
                        VecSetValueLocal(coarse_N[k - 1], i + 2, 0.0, INSERT_VALUES);
                    }
                }

                VecRestoreArray(lcoor, &lcoorp);
                VecAssemblyBegin(coarse_N[k - 1]);
                VecAssemblyEnd(coarse_N[k - 1]);
            }
        }

        // 最细层密度向量将在RestrictDensity中创建

        // 分配上下文数组并保存引用
        mf_ctx_levels = new MatrixFreeContext*[nlvls - 1];
        for (PetscInt k = 0; k < nlvls - 1; k++) {
            MatShellGetContext(coarse_K[k], &mf_ctx_levels[k]);
        }
        // 保存最细层上下文
        MatShellGetContext(K, &mf_ctx);

        // 为每层的KSP设置算子
        for (PetscInt k = 0; k < nlvls; k++) {
            KSP level_ksp;

            if (k == 0) {
                // 最粗层
                PCMGGetCoarseSolve(pc, &level_ksp);
            } else if (k < nlvls - 1) {
                // 中间层
                PCMGGetSmoother(pc, k, &level_ksp);
            } else {
                // 最细层 - 不设置DM，直接使用KSPSetOperators
                continue;
            }

            // 直接设置Shell矩阵作为算子
            ierr = KSPSetOperators(level_ksp, coarse_K[k], coarse_K[k]);
            CHKERRQ(ierr);
            // 不设置DM，避免回调函数被调用
            ierr = KSPSetDMActive(level_ksp, PETSC_FALSE);
            CHKERRQ(ierr);
        }

        // tidy up
        for (PetscInt k = 1; k < nlvls; k++) { // DO NOT DESTROY LEVEL 0
            DMDestroy(&daclist[k]);
        }
        PetscFree(da_list);
        PetscFree(daclist);

        // AVOID THE DEFAULT FOR THE MG PART
        {
            // SET the coarse grid solver:
            // i.e. get a pointer to the ksp and change its settings
            KSP cksp;
            PCMGGetCoarseSolve(pc, &cksp);
            // The solver
            ierr = KSPSetType(cksp, KSPGMRES); // KSPCG, KSPFGMRES
            ierr = KSPGMRESSetRestart(cksp, coarse_restart);
            // ierr = KSPSetType(cksp,KSPCG);

            ierr = KSPSetTolerances(cksp, coarse_rtol, coarse_atol, coarse_dtol, coarse_maxits);
            // The preconditioner - 使用None（Matrix-Free兼容）
            PC cpc;
            KSPGetPC(cksp, &cpc);
            PCSetType(cpc, PCNONE);

            // 允许命令行覆盖
            KSPSetFromOptions(cksp);

            // Set smoothers on all levels (except for coarse grid):
            // 使用Chebyshev + None（不需要矩阵操作）
            for (PetscInt k = 1; k < nlvls; k++) {
                KSP dksp;
                PCMGGetSmoother(pc, k, &dksp);
                PC dpc;
                KSPGetPC(dksp, &dpc);
                ierr = KSPSetType(dksp, KSPCHEBYSHEV);
                ierr = KSPSetTolerances(dksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT,
                                        smooth_sweeps);
                PCSetType(dpc, PCNONE);
                // 允许命令行覆盖
                KSPSetFromOptions(dksp);
            }
        }
    }

    // Write check to screen:
    // Check the overall Krylov solver
    KSPType ksptype;
    KSPGetType(ksp, &ksptype);
    PCType pctype;
    PCGetType(pc, &pctype);
    PetscInt mmax;
    KSPGetTolerances(ksp, NULL, NULL, NULL, &mmax);
    PetscPrintf(PETSC_COMM_WORLD, "##############################################################\n");
    PetscPrintf(PETSC_COMM_WORLD, "################# Linear solver settings #####################\n");
    PetscPrintf(PETSC_COMM_WORLD, "# Main solver: %s, prec.: %s, maxiter.: %i \n", ksptype, pctype, mmax);

    // Only if pcmg is used
    if (pcmg_flag) {
        // Check the smoothers and coarse grid solver:
        for (PetscInt k = 0; k < nlvls; k++) {
            KSP     dksp;
            PC      dpc;
            KSPType dksptype;
            PCMGGetSmoother(pc, k, &dksp);
            KSPGetType(dksp, &dksptype);
            KSPGetPC(dksp, &dpc);
            PCType dpctype;
            PCGetType(dpc, &dpctype);
            PetscInt mmax;
            KSPGetTolerances(dksp, NULL, NULL, NULL, &mmax);
            PetscPrintf(PETSC_COMM_WORLD, "# Level %i smoother: %s, prec.: %s, sweep: %i \n", k, dksptype, dpctype,
                        mmax);
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, "##############################################################\n");

    return (ierr);
}

PetscErrorCode LinearElasticity::DMDAGetElements_3D(DM dm, PetscInt* nel, PetscInt* nen, const PetscInt* e[]) {
    PetscErrorCode ierr;
    DM_DA*         da = (DM_DA*)dm->data;
    PetscInt       i, xs, xe, Xs, Xe;
    PetscInt       j, ys, ye, Ys, Ye;
    PetscInt       k, zs, ze, Zs, Ze;
    PetscInt       cnt = 0, cell[8], ns = 1, nn = 8;
    PetscInt       c;
    if (!da->e) {
        if (da->elementtype == DMDA_ELEMENT_Q1) {
            ns = 1;
            nn = 8;
        }
        ierr = DMDAGetCorners(dm, &xs, &ys, &zs, &xe, &ye, &ze);
        CHKERRQ(ierr);
        ierr = DMDAGetGhostCorners(dm, &Xs, &Ys, &Zs, &Xe, &Ye, &Ze);
        CHKERRQ(ierr);
        xe += xs;
        Xe += Xs;
        if (xs != Xs)
            xs -= 1;
        ye += ys;
        Ye += Ys;
        if (ys != Ys)
            ys -= 1;
        ze += zs;
        Ze += Zs;
        if (zs != Zs)
            zs -= 1;
        da->ne = ns * (xe - xs - 1) * (ye - ys - 1) * (ze - zs - 1);
        PetscMalloc((1 + nn * da->ne) * sizeof(PetscInt), &da->e);
        for (k = zs; k < ze - 1; k++) {
            for (j = ys; j < ye - 1; j++) {
                for (i = xs; i < xe - 1; i++) {
                    cell[0] = (i - Xs) + (j - Ys) * (Xe - Xs) + (k - Zs) * (Xe - Xs) * (Ye - Ys);
                    cell[1] = (i - Xs + 1) + (j - Ys) * (Xe - Xs) + (k - Zs) * (Xe - Xs) * (Ye - Ys);
                    cell[2] = (i - Xs + 1) + (j - Ys + 1) * (Xe - Xs) + (k - Zs) * (Xe - Xs) * (Ye - Ys);
                    cell[3] = (i - Xs) + (j - Ys + 1) * (Xe - Xs) + (k - Zs) * (Xe - Xs) * (Ye - Ys);
                    cell[4] = (i - Xs) + (j - Ys) * (Xe - Xs) + (k - Zs + 1) * (Xe - Xs) * (Ye - Ys);
                    cell[5] = (i - Xs + 1) + (j - Ys) * (Xe - Xs) + (k - Zs + 1) * (Xe - Xs) * (Ye - Ys);
                    cell[6] = (i - Xs + 1) + (j - Ys + 1) * (Xe - Xs) + (k - Zs + 1) * (Xe - Xs) * (Ye - Ys);
                    cell[7] = (i - Xs) + (j - Ys + 1) * (Xe - Xs) + (k - Zs + 1) * (Xe - Xs) * (Ye - Ys);
                    if (da->elementtype == DMDA_ELEMENT_Q1) {
                        for (c = 0; c < ns * nn; c++)
                            da->e[cnt++] = cell[c];
                    }
                }
            }
        }
    }
    *nel = da->ne;
    *nen = nn;
    *e   = da->e;
    return (0);
}

PetscInt LinearElasticity::Hex8Isoparametric(PetscScalar* X, PetscScalar* Y, PetscScalar* Z, PetscScalar nu,
                                             PetscInt redInt, PetscScalar* ke) {
    // HEX8_ISOPARAMETRIC - Computes HEX8 isoparametric element matrices
    // The element stiffness matrix is computed as:
    //
    //       ke = int(int(int(B^T*C*B,x),y),z)
    //
    // For an isoparameteric element this integral becomes:
    //
    //       ke = int(int(int(B^T*C*B*det(J),xi=-1..1),eta=-1..1),zeta=-1..1)
    //
    // where B is the more complicated expression:
    // B = [dx*alpha1 + dy*alpha2 + dz*alpha3]*N
    // where
    // dx = [invJ11 invJ12 invJ13]*[dxi deta dzeta]
    // dy = [invJ21 invJ22 invJ23]*[dxi deta dzeta]
    // dy = [invJ31 invJ32 invJ33]*[dxi deta dzeta]
    //
    // Remark: The elasticity modulus is left out in the below
    // computations, because we multiply with it afterwards (the aim is
    // topology optimization).
    // Furthermore, this is not the most efficient code, but it is readable.
    //
    /////////////////////////////////////////////////////////////////////////////////
    //////// INPUT:
    // X, Y, Z  = Vectors containing the coordinates of the eight nodes
    //               (x1,y1,z1,x2,y2,z2,...,x8,y8,z8). Where node 1 is in the
    //               lower left corner, and node 2 is the next node
    //               counterclockwise (looking in the negative z-dir). Finish the
    //               x-y-plane and then move in the positive z-dir.
    // redInt   = Reduced integration option boolean (here an integer).
    //           	redInt == 0 (false): Full integration
    //           	redInt == 1 (true): Reduced integration
    // nu 		= Poisson's ratio.
    //
    //////// OUTPUT:
    // ke  = Element stiffness matrix. Needs to be multiplied with elasticity
    // modulus
    //
    //   Written 2013 at
    //   Department of Mechanical Engineering
    //   Technical University of Denmark (DTU).
    /////////////////////////////////////////////////////////////////////////////////

    //// COMPUTE ELEMENT STIFFNESS MATRIX
    // Lame's parameters (with E=1.0):
    PetscScalar lambda = nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    PetscScalar mu     = 1.0 / (2.0 * (1.0 + nu));
    // Constitutive matrix
    PetscScalar C[6][6] = {{lambda + 2.0 * mu, lambda, lambda, 0.0, 0.0, 0.0},
                           {lambda, lambda + 2.0 * mu, lambda, 0.0, 0.0, 0.0},
                           {lambda, lambda, lambda + 2.0 * mu, 0.0, 0.0, 0.0},
                           {0.0, 0.0, 0.0, mu, 0.0, 0.0},
                           {0.0, 0.0, 0.0, 0.0, mu, 0.0},
                           {0.0, 0.0, 0.0, 0.0, 0.0, mu}};
    // Gauss points (GP) and weigths
    // Two Gauss points in all directions (total of eight)
    PetscScalar GP[2] = {-0.577350269189626, 0.577350269189626};
    // Corresponding weights
    PetscScalar W[2] = {1.0, 1.0};
    // If reduced integration only use one GP
    if (redInt) {
        GP[0] = 0.0;
        W[0]  = 2.0;
    }
    // Matrices that help when we gather the strain-displacement matrix:
    PetscScalar alpha1[6][3];
    PetscScalar alpha2[6][3];
    PetscScalar alpha3[6][3];
    memset(alpha1, 0, sizeof(alpha1[0][0]) * 6 * 3); // zero out
    memset(alpha2, 0, sizeof(alpha2[0][0]) * 6 * 3); // zero out
    memset(alpha3, 0, sizeof(alpha3[0][0]) * 6 * 3); // zero out
    alpha1[0][0] = 1.0;
    alpha1[3][1] = 1.0;
    alpha1[5][2] = 1.0;
    alpha2[1][1] = 1.0;
    alpha2[3][0] = 1.0;
    alpha2[4][2] = 1.0;
    alpha3[2][2] = 1.0;
    alpha3[4][1] = 1.0;
    alpha3[5][0] = 1.0;
    PetscScalar  dNdxi[8];
    PetscScalar  dNdeta[8];
    PetscScalar  dNdzeta[8];
    PetscScalar  J[3][3];
    PetscScalar  invJ[3][3];
    PetscScalar  beta[6][3];
    PetscScalar  B[6][24]; // Note: Small enough to be allocated on stack
    PetscScalar* dN;
    // Make sure the stiffness matrix is zeroed out:
    memset(ke, 0, sizeof(ke[0]) * 24 * 24);
    // Perform the numerical integration
    for (PetscInt ii = 0; ii < 2 - redInt; ii++) {
        for (PetscInt jj = 0; jj < 2 - redInt; jj++) {
            for (PetscInt kk = 0; kk < 2 - redInt; kk++) {
                // Integration point
                PetscScalar xi   = GP[ii];
                PetscScalar eta  = GP[jj];
                PetscScalar zeta = GP[kk];
                // Differentiated shape functions
                DifferentiatedShapeFunctions(xi, eta, zeta, dNdxi, dNdeta, dNdzeta);
                // Jacobian
                J[0][0] = Dot(dNdxi, X, 8);
                J[0][1] = Dot(dNdxi, Y, 8);
                J[0][2] = Dot(dNdxi, Z, 8);
                J[1][0] = Dot(dNdeta, X, 8);
                J[1][1] = Dot(dNdeta, Y, 8);
                J[1][2] = Dot(dNdeta, Z, 8);
                J[2][0] = Dot(dNdzeta, X, 8);
                J[2][1] = Dot(dNdzeta, Y, 8);
                J[2][2] = Dot(dNdzeta, Z, 8);
                // Inverse and determinant
                PetscScalar detJ = Inverse3M(J, invJ);
                // Weight factor at this point
                PetscScalar weight = W[ii] * W[jj] * W[kk] * detJ;
                // Strain-displacement matrix
                memset(B, 0, sizeof(B[0][0]) * 6 * 24); // zero out
                for (PetscInt ll = 0; ll < 3; ll++) {
                    // Add contributions from the different derivatives
                    if (ll == 0) {
                        dN = dNdxi;
                    }
                    if (ll == 1) {
                        dN = dNdeta;
                    }
                    if (ll == 2) {
                        dN = dNdzeta;
                    }
                    // Assemble strain operator
                    for (PetscInt i = 0; i < 6; i++) {
                        for (PetscInt j = 0; j < 3; j++) {
                            beta[i][j] =
                                invJ[0][ll] * alpha1[i][j] + invJ[1][ll] * alpha2[i][j] + invJ[2][ll] * alpha3[i][j];
                        }
                    }
                    // Add contributions to strain-displacement matrix
                    for (PetscInt i = 0; i < 6; i++) {
                        for (PetscInt j = 0; j < 24; j++) {
                            B[i][j] = B[i][j] + beta[i][j % 3] * dN[j / 3];
                        }
                    }
                }
                // Finally, add to the element matrix
                for (PetscInt i = 0; i < 24; i++) {
                    for (PetscInt j = 0; j < 24; j++) {
                        for (PetscInt k = 0; k < 6; k++) {
                            for (PetscInt l = 0; l < 6; l++) {

                                ke[j + 24 * i] = ke[j + 24 * i] + weight * (B[k][i] * C[k][l] * B[l][j]);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
PetscScalar LinearElasticity::Dot(PetscScalar* v1, PetscScalar* v2, PetscInt l) {
    // Function that returns the dot product of v1 and v2,
    // which must have the same length l
    PetscScalar result = 0.0;
    for (PetscInt i = 0; i < l; i++) {
        result = result + v1[i] * v2[i];
    }
    return result;
}

void LinearElasticity::DifferentiatedShapeFunctions(PetscScalar xi, PetscScalar eta, PetscScalar zeta,
                                                    PetscScalar* dNdxi, PetscScalar* dNdeta, PetscScalar* dNdzeta) {
    // differentiatedShapeFunctions - Computes differentiated shape functions
    // At the point given by (xi, eta, zeta).
    // With respect to xi:
    dNdxi[0] = -0.125 * (1.0 - eta) * (1.0 - zeta);
    dNdxi[1] = 0.125 * (1.0 - eta) * (1.0 - zeta);
    dNdxi[2] = 0.125 * (1.0 + eta) * (1.0 - zeta);
    dNdxi[3] = -0.125 * (1.0 + eta) * (1.0 - zeta);
    dNdxi[4] = -0.125 * (1.0 - eta) * (1.0 + zeta);
    dNdxi[5] = 0.125 * (1.0 - eta) * (1.0 + zeta);
    dNdxi[6] = 0.125 * (1.0 + eta) * (1.0 + zeta);
    dNdxi[7] = -0.125 * (1.0 + eta) * (1.0 + zeta);
    // With respect to eta:
    dNdeta[0] = -0.125 * (1.0 - xi) * (1.0 - zeta);
    dNdeta[1] = -0.125 * (1.0 + xi) * (1.0 - zeta);
    dNdeta[2] = 0.125 * (1.0 + xi) * (1.0 - zeta);
    dNdeta[3] = 0.125 * (1.0 - xi) * (1.0 - zeta);
    dNdeta[4] = -0.125 * (1.0 - xi) * (1.0 + zeta);
    dNdeta[5] = -0.125 * (1.0 + xi) * (1.0 + zeta);
    dNdeta[6] = 0.125 * (1.0 + xi) * (1.0 + zeta);
    dNdeta[7] = 0.125 * (1.0 - xi) * (1.0 + zeta);
    // With respect to zeta:
    dNdzeta[0] = -0.125 * (1.0 - xi) * (1.0 - eta);
    dNdzeta[1] = -0.125 * (1.0 + xi) * (1.0 - eta);
    dNdzeta[2] = -0.125 * (1.0 + xi) * (1.0 + eta);
    dNdzeta[3] = -0.125 * (1.0 - xi) * (1.0 + eta);
    dNdzeta[4] = 0.125 * (1.0 - xi) * (1.0 - eta);
    dNdzeta[5] = 0.125 * (1.0 + xi) * (1.0 - eta);
    dNdzeta[6] = 0.125 * (1.0 + xi) * (1.0 + eta);
    dNdzeta[7] = 0.125 * (1.0 - xi) * (1.0 + eta);
}

PetscScalar LinearElasticity::Inverse3M(PetscScalar J[][3], PetscScalar invJ[][3]) {
    // inverse3M - Computes the inverse of a 3x3 matrix
    PetscScalar detJ = J[0][0] * (J[1][1] * J[2][2] - J[2][1] * J[1][2]) -
                       J[0][1] * (J[1][0] * J[2][2] - J[2][0] * J[1][2]) +
                       J[0][2] * (J[1][0] * J[2][1] - J[2][0] * J[1][1]);
    invJ[0][0] = (J[1][1] * J[2][2] - J[2][1] * J[1][2]) / detJ;
    invJ[0][1] = -(J[0][1] * J[2][2] - J[0][2] * J[2][1]) / detJ;
    invJ[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) / detJ;
    invJ[1][0] = -(J[1][0] * J[2][2] - J[1][2] * J[2][0]) / detJ;
    invJ[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) / detJ;
    invJ[1][2] = -(J[0][0] * J[1][2] - J[0][2] * J[1][0]) / detJ;
    invJ[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) / detJ;
    invJ[2][1] = -(J[0][0] * J[2][1] - J[0][1] * J[2][0]) / detJ;
    invJ[2][2] = (J[0][0] * J[1][1] - J[1][0] * J[0][1]) / detJ;
    return detJ;
}

// 密度限制：从细网格限制到所有粗网格
// 注意：此函数需要正确处理MPI并行环境
PetscErrorCode LinearElasticity::RestrictDensity(Vec xPhys_fine) {
    PetscErrorCode ierr = 0;
    PetscInt rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // 最细层直接使用输入密度
    if (density_levels[nlvls - 1] == NULL) {
        VecDuplicate(xPhys_fine, &density_levels[nlvls - 1]);
    }
    VecCopy(xPhys_fine, density_levels[nlvls - 1]);

    // 从细到粗逐层限制
    for (PetscInt level = nlvls - 2; level >= 0; level--) {
        Vec x_fine = density_levels[level + 1];
        Vec x_coarse = density_levels[level];
        DM dm_fine = (level == nlvls - 2) ? da_nodal : coarse_da[level + 1];
        DM dm_coarse = coarse_da[level];

        // 获取网格尺寸（全局节点数）
        PetscInt M_fine, N_fine, P_fine;
        DMDAGetInfo(dm_fine, NULL, &M_fine, &N_fine, &P_fine, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        PetscInt ne_fine[3] = {M_fine - 1, N_fine - 1, P_fine - 1};

        PetscInt M_coarse, N_coarse, P_coarse;
        DMDAGetInfo(dm_coarse, NULL, &M_coarse, &N_coarse, &P_coarse, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        PetscInt ne_coarse[3] = {M_coarse - 1, N_coarse - 1, P_coarse - 1};

        // 获取本地所有权范围
        PetscInt start_fine, end_fine, start_coarse, end_coarse;
        VecGetOwnershipRange(x_fine, &start_fine, &end_fine);
        VecGetOwnershipRange(x_coarse, &start_coarse, &end_coarse);

        // 对于单进程情况，使用简单的直接访问
        if (size == 1) {
            PetscScalar* xf;
            PetscScalar* xc;
            VecGetArray(x_fine, &xf);
            VecGetArray(x_coarse, &xc);

            for (PetscInt k = 0; k < ne_coarse[2]; k++) {
                for (PetscInt j = 0; j < ne_coarse[1]; j++) {
                    for (PetscInt i = 0; i < ne_coarse[0]; i++) {
                        PetscInt idx_coarse = i + j * ne_coarse[0] + k * ne_coarse[0] * ne_coarse[1];

                        PetscScalar sum = 0.0;
                        PetscInt count = 0;
                        for (PetscInt kk = 0; kk < 2; kk++) {
                            for (PetscInt jj = 0; jj < 2; jj++) {
                                for (PetscInt ii = 0; ii < 2; ii++) {
                                    PetscInt i_fine = 2 * i + ii;
                                    PetscInt j_fine = 2 * j + jj;
                                    PetscInt k_fine = 2 * k + kk;

                                    if (i_fine < ne_fine[0] && j_fine < ne_fine[1] && k_fine < ne_fine[2]) {
                                        PetscInt idx_fine = i_fine + j_fine * ne_fine[0] + k_fine * ne_fine[0] * ne_fine[1];
                                        sum += xf[idx_fine];
                                        count++;
                                    }
                                }
                            }
                        }
                        xc[idx_coarse] = (count > 0) ? (sum / count) : 0.5;
                    }
                }
            }

            VecRestoreArray(x_fine, &xf);
            VecRestoreArray(x_coarse, &xc);
        } else {
            // 多进程情况：使用VecScatter进行通信
            // 首先，每个进程计算其本地粗网格单元需要的细网格单元
            PetscInt local_coarse_size = end_coarse - start_coarse;

            // 创建临时数组存储结果
            PetscScalar* local_coarse_vals;
            PetscMalloc1(local_coarse_size, &local_coarse_vals);

            // 获取细网格的全局数据（通过scatter）
            Vec x_fine_seq;
            VecScatter scatter;
            VecScatterCreateToAll(x_fine, &scatter, &x_fine_seq);
            VecScatterBegin(scatter, x_fine, x_fine_seq, INSERT_VALUES, SCATTER_FORWARD);
            VecScatterEnd(scatter, x_fine, x_fine_seq, INSERT_VALUES, SCATTER_FORWARD);

            PetscScalar* xf_all;
            VecGetArray(x_fine_seq, &xf_all);

            // 计算本地粗网格单元的密度
            for (PetscInt idx_local = 0; idx_local < local_coarse_size; idx_local++) {
                PetscInt idx_global = start_coarse + idx_local;

                // 将全局索引转换为(i,j,k)
                PetscInt k = idx_global / (ne_coarse[0] * ne_coarse[1]);
                PetscInt rem = idx_global % (ne_coarse[0] * ne_coarse[1]);
                PetscInt j = rem / ne_coarse[0];
                PetscInt i = rem % ne_coarse[0];

                PetscScalar sum = 0.0;
                PetscInt count = 0;
                for (PetscInt kk = 0; kk < 2; kk++) {
                    for (PetscInt jj = 0; jj < 2; jj++) {
                        for (PetscInt ii = 0; ii < 2; ii++) {
                            PetscInt i_fine = 2 * i + ii;
                            PetscInt j_fine = 2 * j + jj;
                            PetscInt k_fine = 2 * k + kk;

                            if (i_fine < ne_fine[0] && j_fine < ne_fine[1] && k_fine < ne_fine[2]) {
                                PetscInt idx_fine = i_fine + j_fine * ne_fine[0] + k_fine * ne_fine[0] * ne_fine[1];
                                sum += xf_all[idx_fine];
                                count++;
                            }
                        }
                    }
                }
                local_coarse_vals[idx_local] = (count > 0) ? (sum / count) : 0.5;
            }

            VecRestoreArray(x_fine_seq, &xf_all);
            VecScatterDestroy(&scatter);
            VecDestroy(&x_fine_seq);

            // 将结果写入粗网格向量
            PetscScalar* xc;
            VecGetArray(x_coarse, &xc);
            for (PetscInt idx_local = 0; idx_local < local_coarse_size; idx_local++) {
                xc[idx_local] = local_coarse_vals[idx_local];
            }
            VecRestoreArray(x_coarse, &xc);

            PetscFree(local_coarse_vals);
        }
    }

    return ierr;
}

// 计算固定自由度（边界条件）- 提取自SetUpLoadAndBC的逻辑
PetscErrorCode LinearElasticity::ComputeFixedDOFs_Level(DM dm, IS* fixed_is) {
    PetscErrorCode ierr;
    
    // 获取坐标
    Vec lcoor;
    ierr = DMGetCoordinatesLocal(dm, &lcoor);
    CHKERRQ(ierr);
    
    PetscScalar* lcoorp;
    VecGetArray(lcoor, &lcoorp);
    
    // 获取局部向量大小
    PetscInt nn;
    VecGetSize(lcoor, &nn);
    
    // 获取域边界
    PetscScalar xmin = xc[0], xmax = xc[1];
    PetscScalar ymin = xc[2], ymax = xc[3];
    PetscScalar zmin = xc[4], zmax = xc[5];
    
    // 计算epsilon用于浮点比较
    PetscScalar dx = (xmax - xmin) / ne[0];
    PetscScalar dy = (ymax - ymin) / ne[1];
    PetscScalar dz = (zmax - zmin) / ne[2];
    PetscScalar epsi = PetscMin(dx * 0.05, PetscMin(dy * 0.05, dz * 0.05));
    
    // 收集固定的DOF索引
    std::vector<PetscInt> fixed_dofs;
    
    for (PetscInt i = 0; i < nn; i++) {
        // 检查是否在固定边界上（x = xmin的墙面，所有3个DOF都固定）
        if (i % 3 == 0 && PetscAbsScalar(lcoorp[i] - xmin) < epsi) {
            // 这是一个在固定墙上的节点，固定所有3个DOF
            PetscInt node_start = i;
            
            // 将局部索引转换为全局索引
            // 注意：lcoor是局部向量，我们需要全局索引
            // 我们需要使用DM的局部到全局映射
            
            // 简化方法：直接使用局部索引，稍后转换
            fixed_dofs.push_back(node_start);     // u
            fixed_dofs.push_back(node_start + 1); // v
            fixed_dofs.push_back(node_start + 2); // w
        }
    }
    
    VecRestoreArray(lcoor, &lcoorp);
    
    // 创建Index Set
    // 注意：我们需要将局部索引转换为全局索引
    // 使用DM的局部到全局映射
    ISLocalToGlobalMapping ltog;
    ierr = DMGetLocalToGlobalMapping(dm, &ltog);
    CHKERRQ(ierr);
    
    // 转换局部索引到全局索引
    std::vector<PetscInt> global_fixed_dofs(fixed_dofs.size());
    ierr = ISLocalToGlobalMappingApply(ltog, fixed_dofs.size(), fixed_dofs.data(), global_fixed_dofs.data());
    CHKERRQ(ierr);
    
    // 创建IS
    ierr = ISCreateGeneral(PETSC_COMM_WORLD, global_fixed_dofs.size(), global_fixed_dofs.data(), PETSC_COPY_VALUES, fixed_is);
    CHKERRQ(ierr);

    return ierr;
}

// ============================================================================
// Matrix-Free: 创建Shell矩阵
// ============================================================================
PetscErrorCode LinearElasticity::CreateShellMatrix(DM da, Vec xPhys, Mat* A,
                                                    PetscScalar Emin, PetscScalar Emax,
                                                    PetscScalar penal, PetscInt level) {
    PetscErrorCode ierr;

    // 获取矩阵大小
    PetscInt M, N_nodes, P;
    DMDAGetInfo(da, NULL, &M, &N_nodes, &P, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);

    // 创建上下文
    MatrixFreeContext* ctx = new MatrixFreeContext;
    ctx->le = this;
    ctx->xPhys = xPhys;
    ctx->Emin = Emin;
    ctx->Emax = Emax;
    ctx->penal = penal;
    ctx->da = da;
    ctx->level = level;
    ctx->gpu_res = NULL;

    // 获取本地和全局大小
    Vec tmp;
    DMCreateGlobalVector(da, &tmp);
    PetscInt local_size, global_size;
    VecGetLocalSize(tmp, &local_size);
    VecGetSize(tmp, &global_size);
    VecDestroy(&tmp);

    // 获取单元信息用于GPU初始化
    PetscInt nel, nen;
    const PetscInt* necon;
    ierr = DMDAGetElements_3D(da, &nel, &nen, &necon); CHKERRQ(ierr);

    // 获取局部向量大小（含ghost）
    Vec tmp_local;
    DMCreateLocalVector(da, &tmp_local);
    PetscInt local_size_with_ghost;
    VecGetSize(tmp_local, &local_size_with_ghost);
    VecDestroy(&tmp_local);

    // 初始化GPU资源
    ierr = MatrixFreeGPU_Init(&ctx->gpu_res, KE, necon, nel,
                              local_size_with_ghost, local_size); CHKERRQ(ierr);

    DMDARestoreElements(da, &nel, &nen, &necon);

    // 创建Shell矩阵
    ierr = MatCreateShell(PETSC_COMM_WORLD, local_size, local_size,
                          global_size, global_size, ctx, A); CHKERRQ(ierr);

    // 设置矩阵-向量乘积函数
    ierr = MatShellSetOperation(*A, MATOP_MULT, (void(*)(void))MatMult_MatrixFree); CHKERRQ(ierr);

    // 设置获取对角线函数（用于Jacobi预条件器）
    ierr = MatShellSetOperation(*A, MATOP_GET_DIAGONAL, (void(*)(void))MatGetDiagonal_MatrixFree); CHKERRQ(ierr);

    // 设置销毁函数（清理上下文）
    ierr = MatShellSetOperation(*A, MATOP_DESTROY, (void(*)(void))MatDestroy_MatrixFree); CHKERRQ(ierr);

    return 0;
}