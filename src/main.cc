#include "Filter.h"
#include "LinearElasticity.h"
#include "MMA.h"
#include "OC.h"
#include "MPIIO.h"
#include "TopOpt.h"
#include "mpi.h"
#include <petsc.h>
/*
Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013

Updated: June 2019, Niels Aage
Copyright (C) 2013-2019,

Disclaimer:
The authors reserves all rights but does not guaranty that the code is
free from errors. Furthermore, we shall not be liable in any event
caused by the use of the program.
 */

static char help[] = "3D拓扑优化 - 使用PCG求解器\n"
                     "用法: mpirun -np <ngpu> ./topopt -nx <X> -ny <Y> -nz <Z> [选项]\n"
                     "必需参数:\n"
                     "  -nx, -ny, -nz       网格尺寸（节点数）\n"
                     "可选参数:\n"
                     "  -maxItr <200>       最大迭代次数（默认200）\n"
                     "  -use_mma            使用MMA优化器（默认使用OC）\n"
                     "  -output_final_vtk   输出最终VTK文件\n";

// 设置GPU运行环境
static PetscErrorCode SetupGPUEnvironment(int ngpu) {
    PetscErrorCode ierr;

    // 设置向量类型为CUDA
    ierr = PetscOptionsSetValue(NULL, "-dm_vec_type", "cuda"); CHKERRQ(ierr);

    // 设置PCG求解器
    ierr = PetscOptionsSetValue(NULL, "-ksp_type", "cg"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL, "-ksp_rtol", "1e-5"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL, "-ksp_max_it", "1000"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL, "-pc_type", "jacobi"); CHKERRQ(ierr);

    // 默认使用Matrix-Free
    ierr = PetscOptionsSetValue(NULL, "-use_matrix_free", ""); CHKERRQ(ierr);

    if (ngpu > 1) {
        ierr = PetscOptionsSetValue(NULL, "-use_gpu_aware_mpi", "0"); CHKERRQ(ierr);
    }

    return 0;
}

int main(int argc, char* argv[]) {

    // Error code for debugging
    PetscErrorCode ierr;

    // Initialize PETSc / MPI and pass input arguments to PETSc
    PetscInitialize(&argc, &argv, PETSC_NULL, help);

    // 获取MPI进程数作为默认GPU数
    PetscMPIInt mpi_size;
    MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size);

    // 获取ngpu参数（如果指定）
    PetscInt ngpu = mpi_size;
    PetscOptionsGetInt(NULL, NULL, "-ngpu", &ngpu, NULL);

    // 设置GPU环境
    ierr = SetupGPUEnvironment(ngpu); CHKERRQ(ierr);

    // 打印GPU配置信息
    PetscPrintf(PETSC_COMM_WORLD, "# 使用 %d 个GPU运行\n", mpi_size);

    // STEP 1: THE OPTIMIZATION PARAMETERS, DATA AND MESH (!!! THE DMDA !!!)
    TopOpt* opt = new TopOpt();

    // STEP 2: THE PHYSICS
    LinearElasticity* physics = new LinearElasticity(opt->da_nodes);

    // STEP 3: THE FILTERING
    Filter* filter = new Filter(opt->da_nodes, opt->xPhys, opt->filter, opt->rmin);

    // STEP 4: VISUALIZATION USING VTK
    MPIIO* output = new MPIIO(opt->da_nodes, 3, "ux, uy, uz", 3, "x, xTilde, xPhys");
    
    // STEP 5: 优化器 - OC或MMA
    PetscBool use_mma = PETSC_FALSE;
    PetscBool flg;
    PetscOptionsGetBool(NULL, NULL, "-use_mma", &use_mma, &flg);
    
    MMA* mma = NULL;
    OC* oc = NULL;
    PetscInt itr = 0;
    
    if (use_mma) {
        PetscPrintf(PETSC_COMM_WORLD, "# 使用MMA优化器 (Method of Moving Asymptotes)\n");
        opt->AllocateMMAwithRestart(&itr, &mma);
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "# 使用OC优化器 (Optimality Criteria)\n");
        oc = new OC(opt->n, opt->x);
    }

    // VTK输出控制选项
    PetscBool output_final_vtk = PETSC_FALSE;  // 默认不输出VTK
    PetscOptionsGetBool(NULL, NULL, "-output_final_vtk", &output_final_vtk, &flg);

    // STEP 6: FILTER THE INITIAL DESIGN/RESTARTED DESIGN
    ierr = filter->FilterProject(opt->x, opt->xTilde, opt->xPhys, opt->projectionFilter, opt->beta, opt->eta);
    CHKERRQ(ierr);

    // STEP 7: 优化循环
    PetscScalar ch = 1.0;
    double      t1, t2;
    while (itr < opt->maxItr && ch > 0.02) {
        // Update iteration counter
        itr++;

        // start timer
        t1 = MPI_Wtime();

        // Compute (a) obj+const, (b) sens, (c) obj+const+sens
        ierr = physics->ComputeObjectiveConstraintsSensitivities(&(opt->fx), &(opt->gx[0]), opt->dfdx, opt->dgdx[0],
                                                                 opt->xPhys, opt->Emin, opt->Emax, opt->penal,
                                                                 opt->volfrac);
        CHKERRQ(ierr);

        // Compute objective scale
        if (itr == 1) {
            opt->fscale = 10.0 / opt->fx;
        }
        // Scale objectie and sens
        opt->fx = opt->fx * opt->fscale;
        VecScale(opt->dfdx, opt->fscale);

        // Filter sensitivities (chainrule)
        ierr = filter->Gradients(opt->x, opt->xTilde, opt->dfdx, opt->m, opt->dgdx, opt->projectionFilter, opt->beta,
                                 opt->eta);
        CHKERRQ(ierr);

        // 更新设计变量
        if (use_mma) {
            // 设置设计变量的外部移动限制
            ierr = mma->SetOuterMovelimit(opt->Xmin, opt->Xmax, opt->movlim, opt->x, opt->xmin, opt->xmax);
            CHKERRQ(ierr);
            // 使用MMA更新设计
            ierr = mma->Update(opt->x, opt->dfdx, opt->gx, opt->dgdx, opt->xmin, opt->xmax);
            CHKERRQ(ierr);
            // 计算设计变化的无穷范数
            ch = mma->DesignChange(opt->x, opt->xold);
        } else {
            // 设置设计变量的外部移动限制
            ierr = oc->SetOuterMovelimit(opt->Xmin, opt->Xmax, opt->movlim, opt->x, opt->xmin, opt->xmax);
            CHKERRQ(ierr);
            // 使用OC更新设计
            ierr = oc->Update(opt->x, opt->dfdx, opt->gx, opt->dgdx, opt->xmin, opt->xmax);
            CHKERRQ(ierr);
            // 计算设计变化的无穷范数
            ch = oc->DesignChange(opt->x, opt->xold);
        }

        // Increase beta if needed
        PetscBool changeBeta = PETSC_FALSE;
        if (opt->projectionFilter) {
            changeBeta = filter->IncreaseBeta(&(opt->beta), opt->betaFinal, opt->gx[0], itr, ch);
        }

        // Filter design field
        ierr = filter->FilterProject(opt->x, opt->xTilde, opt->xPhys, opt->projectionFilter, opt->beta, opt->eta);
        CHKERRQ(ierr);

        // Discreteness measure
        PetscScalar mnd = filter->GetMND(opt->xPhys);

        // 停止计时
        t2 = MPI_Wtime();

        // 每10次迭代输出一次
        if (itr % 10 == 0) {
            PetscPrintf(PETSC_COMM_WORLD,
                        "It.: %i, 真实fx: %f, 缩放fx: %f, gx[0]: %f, ch.: %f, "
                        "mnd.: %f, KSP: %d, 总KSP: %d, time: %f\n",
                        itr, opt->fx / opt->fscale, opt->fx, opt->gx[0], ch, mnd,
                        physics->GetLastKSPIterations(), physics->GetTotalKSPIterations(), t2 - t1);
        }
    }
    
    // 输出最终结果
    PetscPrintf(PETSC_COMM_WORLD, "\n优化完成！\n");
    PetscPrintf(PETSC_COMM_WORLD, "总迭代次数: %d\n", itr);
    PetscPrintf(PETSC_COMM_WORLD, "最终目标函数: %f\n", opt->fx / opt->fscale);
    PetscPrintf(PETSC_COMM_WORLD, "最终体积约束: %f\n", opt->gx[0]);
    PetscPrintf(PETSC_COMM_WORLD, "总KSP迭代次数: %d\n", physics->GetTotalKSPIterations());

    // 输出最终VTK文件（如果指定）
    if (output_final_vtk) {
        PetscPrintf(PETSC_COMM_WORLD, "正在输出最终VTK文件...\n");
        output->WriteVTK(physics->da_nodal, physics->GetStateField(), opt->x, opt->xTilde, opt->xPhys, itr);
    }

    // STEP 8: 清理
    if (use_mma && mma != NULL) {
        delete mma;
    }
    if (!use_mma && oc != NULL) {
        delete oc;
    }
    delete output;
    delete filter;
    delete opt;
    delete physics;

    // Finalize PETSc / MPI
    PetscFinalize();
    return 0;
}
