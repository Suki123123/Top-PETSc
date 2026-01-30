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

static char help[] = "3D TopOpt using KSP-MG on PETSc's DMDA (structured grids) \n"
                     "Usage: mpirun -np <ngpu> ./topopt [options]\n"
                     "Options:\n"
                     "  -ngpu <1|2>         Number of GPUs (default: auto from MPI)\n"
                     "  -output_vtk <0|1>   Output intermediate VTK files (default: 1)\n"
                     "  -output_final <0|1> Output final VTK file (default: 1)\n"
                     "  -use_oc             Use OC optimizer (default)\n"
                     "  -use_mma            Use MMA optimizer\n"
                     "  -nx, -ny, -nz       Mesh size\n"
                     "  -volfrac            Volume fraction\n"
                     "  -maxItr             Maximum iterations\n";

// 设置GPU运行环境
static PetscErrorCode SetupGPUEnvironment(int ngpu) {
    PetscErrorCode ierr;

    // 设置向量类型为CUDA
    ierr = PetscOptionsSetValue(NULL, "-dm_vec_type", "cuda"); CHKERRQ(ierr);

    // 设置求解器
    ierr = PetscOptionsSetValue(NULL, "-ksp_type", "cg"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL, "-ksp_rtol", "1e-5"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL, "-ksp_max_it", "500"); CHKERRQ(ierr);
    ierr = PetscOptionsSetValue(NULL, "-pc_type", "jacobi"); CHKERRQ(ierr);

    // 默认使用OC优化器
    ierr = PetscOptionsSetValue(NULL, "-use_oc", ""); CHKERRQ(ierr);

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
    PetscPrintf(PETSC_COMM_WORLD, "# Running with %d GPU(s)\n", mpi_size);

    // STEP 1: THE OPTIMIZATION PARAMETERS, DATA AND MESH (!!! THE DMDA !!!)
    TopOpt* opt = new TopOpt();

    // STEP 2: THE PHYSICS
    LinearElasticity* physics = new LinearElasticity(opt->da_nodes);

    // STEP 3: THE FILTERING
    Filter* filter = new Filter(opt->da_nodes, opt->xPhys, opt->filter, opt->rmin);

    // STEP 4: VISUALIZATION USING VTK
    MPIIO* output = new MPIIO(opt->da_nodes, 3, "ux, uy, uz", 3, "x, xTilde, xPhys");
    // STEP 5: THE OPTIMIZER MMA or OC
    PetscBool use_oc = PETSC_TRUE;   // 默认使用OC
    PetscBool use_mma = PETSC_FALSE;
    PetscBool flg_oc, flg_mma;
    PetscOptionsGetBool(NULL, NULL, "-use_oc", &use_oc, &flg_oc);
    PetscOptionsGetBool(NULL, NULL, "-use_mma", &use_mma, &flg_mma);

    // 如果指定了-use_mma，则使用MMA
    if (flg_mma && use_mma) {
        use_oc = PETSC_FALSE;
    }

    // VTK输出控制选项
    PetscBool output_vtk = PETSC_TRUE;  // 默认输出VTK
    PetscBool output_final_vtk = PETSC_TRUE;  // 默认输出最终VTK
    PetscInt vtk_interval = 20;  // VTK输出间隔
    PetscOptionsGetBool(NULL, NULL, "-output_vtk", &output_vtk, NULL);
    PetscOptionsGetBool(NULL, NULL, "-output_final", &output_final_vtk, NULL);
    PetscOptionsGetInt(NULL, NULL, "-vtk_interval", &vtk_interval, NULL);

    MMA*     mma = NULL;
    OC*      oc = NULL;
    PetscInt itr = 0;

    if (use_oc) {
        PetscPrintf(PETSC_COMM_WORLD, "# Using OC (Optimality Criteria) optimizer\n");
        oc = new OC(opt->n, opt->x);
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "# Using MMA (Method of Moving Asymptotes) optimizer\n");
        opt->AllocateMMAwithRestart(&itr, &mma); // allow for restart !
    }
    // mma->SetAsymptotes(0.2, 0.65, 1.05);

    // STEP 6: FILTER THE INITIAL DESIGN/RESTARTED DESIGN
    ierr = filter->FilterProject(opt->x, opt->xTilde, opt->xPhys, opt->projectionFilter, opt->beta, opt->eta);
    CHKERRQ(ierr);

    // STEP 7: OPTIMIZATION LOOP
    PetscScalar ch = 1.0;
    double      t1, t2;
    while (itr < opt->maxItr && ch > 0.01) {
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

        // Sets outer movelimits on design variables
        if (use_oc) {
            ierr = oc->SetOuterMovelimit(opt->Xmin, opt->Xmax, opt->movlim, opt->x, opt->xmin, opt->xmax);
            CHKERRQ(ierr);
            // Update design by OC
            ierr = oc->Update(opt->x, opt->dfdx, opt->gx, opt->dgdx, opt->xmin, opt->xmax);
            CHKERRQ(ierr);
            // Inf norm on the design change
            ch = oc->DesignChange(opt->x, opt->xold);
        } else {
            ierr = mma->SetOuterMovelimit(opt->Xmin, opt->Xmax, opt->movlim, opt->x, opt->xmin, opt->xmax);
            CHKERRQ(ierr);
            // Update design by MMA
            ierr = mma->Update(opt->x, opt->dfdx, opt->gx, opt->dgdx, opt->xmin, opt->xmax);
            CHKERRQ(ierr);
            // Inf norm on the design change
            ch = mma->DesignChange(opt->x, opt->xold);
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

        // stop timer
        t2 = MPI_Wtime();

        // Print to screen
        PetscPrintf(PETSC_COMM_WORLD,
                    "It.: %i, True fx: %f, Scaled fx: %f, gx[0]: %f, ch.: %f, "
                    "mnd.: %f, time: %f\n",
                    itr, opt->fx / opt->fscale, opt->fx, opt->gx[0], ch, mnd, t2 - t1);

        // Write field data: first 10 iterations and then every vtk_interval
        if (output_vtk && (itr < 11 || itr % vtk_interval == 0 || changeBeta)) {
            output->WriteVTK(physics->da_nodal, physics->GetStateField(), opt->x, opt->xTilde, opt->xPhys, itr);
        }

        // Dump data needed for restarting code at termination
        if (itr % 10 == 0 && !use_oc) {
            opt->WriteRestartFiles(&itr, mma);
            physics->WriteRestartFiles();
        }
    }
    // Write restart WriteRestartFiles
    if (!use_oc) {
        opt->WriteRestartFiles(&itr, mma);
    }
    physics->WriteRestartFiles();

    // Dump final design
    if (output_final_vtk) {
        output->WriteVTK(physics->da_nodal, physics->GetStateField(), opt->x, opt->xTilde, opt->xPhys, itr + 1);
    }

    // STEP 7: CLEAN UP AFTER YOURSELF
    if (mma) delete mma;
    if (oc) delete oc;
    delete output;
    delete filter;
    delete opt;
    delete physics;

    // Finalize PETSc / MPI
    PetscFinalize();
    return 0;
}
