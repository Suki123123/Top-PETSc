#include "LinearElasticity.h"
#include <vector>

/*
 Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013

 Disclaimer:
 The authors reserves all rights but does not guaranty that the code is
 free from errors. Furthermore, we shall not be liable in any event
 caused by the use of the program.
*/

// ============================================================================
// 回调函数上下文结构（用于KSPSetComputeOperators）
// ============================================================================
typedef struct {
    LinearElasticity* le;      // LinearElasticity对象指针
    PetscInt          level;   // 当前层级
} MGLevelContext;

// ============================================================================
// 回调函数：为指定层组装刚度矩阵
// ============================================================================
static PetscErrorCode ComputeMatrix_Level(KSP ksp, Mat J, Mat P, void* ctx) {
    PetscErrorCode ierr;
    MGLevelContext* level_ctx = (MGLevelContext*)ctx;
    LinearElasticity* le = level_ctx->le;
    PetscInt level = level_ctx->level;
    
    // 获取该层的DM、密度向量和Dirichlet向量
    DM dm_level;
    KSPGetDM(ksp, &dm_level);
    
    Vec xPhys_level = le->density_levels[level];
    Vec N_level = (level == le->nlvls - 1) ? le->N : le->coarse_N[level];
    
    // 检查密度向量是否有效
    if (xPhys_level == NULL) {
        PetscPrintf(PETSC_COMM_WORLD, "ERROR: density_levels[%d] is NULL!\n", level);
        return PETSC_ERR_ARG_NULL;
    }
    
    // 组装该层的刚度矩阵
    ierr = le->AssembleStiffnessMatrix_Level(level, dm_level, xPhys_level, P, N_level,
                                              le->current_Emin, le->current_Emax, le->current_penal);
    CHKERRQ(ierr);
    
    // 如果J != P，也需要组装J
    if (J != P) {
        ierr = le->AssembleStiffnessMatrix_Level(level, dm_level, xPhys_level, J, N_level,
                                                  le->current_Emin, le->current_Emax, le->current_penal);
        CHKERRQ(ierr);
    }
    
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

    // Parameters - to be changed on read of variables
    nu    = 0.3;
    nlvls = 4;
    PetscBool flg;
    PetscOptionsGetInt(NULL, NULL, "-nlvls", &nlvls, &flg);
    PetscOptionsGetReal(NULL, NULL, "-nu", &nu, &flg);

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
    // In this case: N = the wall at x=xmin is fully clamped
    //               RHS(z) = sin(pi*y/Ly) at x=xmax,z=zmin;
    // OR
    //               RHS(z) = -0.1 at x=xmax,z=zmin;
    PetscScalar LoadIntensity = -0.001;
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

    // Assemble the stiffness matrix (最细层)
    ierr = AssembleStiffnessMatrix(xPhys, Emin, Emax, penal);
    CHKERRQ(ierr);

    // Setup the solver
    if (ksp == NULL) {
        ierr = SetUpSolver();
        CHKERRQ(ierr);
        
        // 如果使用几何重离散化，在第一次solve之前初始化密度
        if (use_geometric_mg) {
            current_Emin = Emin;
            current_Emax = Emax;
            current_penal = penal;
            
            // 初始化密度向量
            ierr = RestrictDensity(xPhys);
            CHKERRQ(ierr);
        }
    } else {
        // 如果使用几何重离散化，更新材料参数和密度
        if (use_geometric_mg) {
            current_Emin = Emin;
            current_Emax = Emax;
            current_penal = penal;
            
            // 密度限制：从细网格限制到所有粗网格
            ierr = RestrictDensity(xPhys);
            CHKERRQ(ierr);
        }
        
        ierr = KSPSetOperators(ksp, K, K);
        CHKERRQ(ierr);
        KSPSetUp(ksp);
    }

    // Solve (回调函数会自动被调用)
    ierr = KSPSolve(ksp, RHS, U);
    CHKERRQ(ierr);

    // DEBUG
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
    // DMDAGetElements(da_nodes,&nel,&nen,&necon); // Still issue with elemtype
    // change !

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
        // Set the Senstivity
        df[i] = -1.0 * penal * PetscPowScalar(xp[i], penal - 1) * (Emax - Emin) * uKu;
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

    VecRestoreArray(xPhys, &xp);
    VecRestoreArray(Uloc, &up);
    VecRestoreArray(dfdx, &df);
    VecDestroy(&Uloc);

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

PetscErrorCode LinearElasticity::AssembleStiffnessMatrix(Vec xPhys, PetscScalar Emin, PetscScalar Emax,
                                                         PetscScalar penal) {

    PetscErrorCode ierr;

    // Get the FE mesh structure (from the nodal mesh)
    PetscInt        nel, nen;
    const PetscInt* necon;
    ierr = DMDAGetElements_3D(da_nodal, &nel, &nen, &necon);
    CHKERRQ(ierr);

    // Get pointer to the densities
    PetscScalar* xp;
    VecGetArray(xPhys, &xp);

    // Zero the matrix
    MatZeroEntries(K);

    // Edof array
    PetscInt    edof[24];
    PetscScalar ke[24 * 24];

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
        PetscScalar dens = Emin + PetscPowScalar(xp[i], penal) * (Emax - Emin);
        for (PetscInt k = 0; k < 24 * 24; k++) {
            ke[k] = KE[k] * dens;
        }
        // Add values to the sparse matrix
        ierr = MatSetValuesLocal(K, 24, edof, 24, edof, ke, ADD_VALUES);
        CHKERRQ(ierr);
    }
    MatAssemblyBegin(K, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K, MAT_FINAL_ASSEMBLY);

    // Impose the dirichlet conditions, i.e. K = N'*K*N - (N-I)
    // 1.: K = N'*K*N
    MatDiagonalScale(K, N, N);
    // 2. Add ones, i.e. K = K + NI, NI = I - N
    Vec NI;
    VecDuplicate(N, &NI);
    VecSet(NI, 1.0);
    VecAXPY(NI, -1.0, N);
    MatDiagonalSet(K, NI, ADD_VALUES);

    // Zero out possible loads in the RHS that coincide
    // with Dirichlet conditions
    VecPointwiseMult(RHS, RHS, N);

    VecDestroy(&NI);
    VecRestoreArray(xPhys, &xp);
    DMDARestoreElements(da_nodal, &nel, &nen, &necon);

    return ierr;
}

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
        Mat R;

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
            
            // 为粗网格创建刚度矩阵
            DMCreateMatrix(coarse_da[k - 1], &coarse_K[k - 1]);
            
            // 为粗网格创建Dirichlet向量
            DMCreateGlobalVector(coarse_da[k - 1], &coarse_N[k - 1]);
            VecSet(coarse_N[k - 1], 1.0);
            
            // 为粗网格创建密度向量（初始化为0）
            DMCreateGlobalVector(coarse_da[k - 1], &density_levels[k - 1]);
            VecSet(density_levels[k - 1], 0.0);
        }
        
        // 最细层密度向量将在RestrictDensity中创建
        
        // 为粗网格DM注册回调函数
        for (PetscInt k = 0; k < nlvls - 1; k++) {
            MGLevelContext* level_ctx = new MGLevelContext;
            level_ctx->le = this;
            level_ctx->level = k;
            
            ierr = DMKSPSetComputeOperators(coarse_da[k], ComputeMatrix_Level, level_ctx);
            CHKERRQ(ierr);
        }
        
        // 为每层的KSP设置DM
        for (PetscInt k = 0; k < nlvls; k++) {
            KSP level_ksp;
            DM dm_level;
            
            if (k == 0) {
                // 最粗层
                PCMGGetCoarseSolve(pc, &level_ksp);
                dm_level = coarse_da[0];
            } else if (k < nlvls - 1) {
                // 中间层
                PCMGGetSmoother(pc, k, &level_ksp);
                dm_level = coarse_da[k];
            } else {
                // 最细层 - 不设置DM，直接使用KSPSetOperators
                continue;
            }
            
            // 将DM与KSP关联
            ierr = KSPSetDM(level_ksp, dm_level);
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
            // The preconditioner
            PC cpc;
            KSPGetPC(cksp, &cpc);
            PCSetType(cpc, PCSOR); // PCGAMG, PCSOR, PCSPAI (NEEDS TO BE COMPILED), PCJACOBI

            // Set smoothers on all levels (except for coarse grid):
            for (PetscInt k = 1; k < nlvls; k++) {
                KSP dksp;
                PCMGGetSmoother(pc, k, &dksp);
                PC dpc;
                KSPGetPC(dksp, &dpc);
                ierr = KSPSetType(dksp,
                                  KSPGMRES); // KSPCG, KSPGMRES, KSPCHEBYSHEV (VERY GOOD FOR SPD)
                ierr = KSPGMRESSetRestart(dksp, smooth_sweeps);
                // ierr = KSPSetType(dksp,KSPCHEBYSHEV);
                ierr = KSPSetTolerances(dksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT,
                                        smooth_sweeps); // NOTE in the above maxitr=restart;
                PCSetType(dpc, PCSOR);                  // PCJACOBI, PCSOR for KSPCHEBYSHEV very good
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
PetscErrorCode LinearElasticity::RestrictDensity(Vec xPhys_fine) {
    PetscErrorCode ierr = 0;
    
    // 最细层直接使用输入密度
    if (density_levels[nlvls - 1] == NULL) {
        VecDuplicate(xPhys_fine, &density_levels[nlvls - 1]);
    }
    VecCopy(xPhys_fine, density_levels[nlvls - 1]);
    
    // 从细到粗逐层限制
    // 注意：密度是基于单元的，而插值算子是基于节点的，所以不能直接使用
    // 我们使用简单的平均方法：每个粗单元的密度 = 对应8个细单元密度的平均
    for (PetscInt level = nlvls - 2; level >= 0; level--) {
        Vec x_fine = density_levels[level + 1];
        Vec x_coarse = density_levels[level];
        DM dm_fine = (level == nlvls - 2) ? da_nodal : coarse_da[level + 1];
        DM dm_coarse = coarse_da[level];
        
        // 获取网格尺寸（节点数）
        PetscInt M_fine, N_fine, P_fine;
        DMDAGetInfo(dm_fine, NULL, &M_fine, &N_fine, &P_fine, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        PetscInt ne_fine[3] = {M_fine - 1, N_fine - 1, P_fine - 1};  // 单元数
        
        PetscInt M_coarse, N_coarse, P_coarse;
        DMDAGetInfo(dm_coarse, NULL, &M_coarse, &N_coarse, &P_coarse, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
        PetscInt ne_coarse[3] = {M_coarse - 1, N_coarse - 1, P_coarse - 1};  // 单元数
        
        // 获取密度数组（全局）
        PetscScalar* xf;
        PetscScalar* xc;
        VecGetArray(x_fine, &xf);
        VecGetArray(x_coarse, &xc);
        
        // 获取向量大小
        PetscInt size_fine, size_coarse;
        VecGetLocalSize(x_fine, &size_fine);
        VecGetLocalSize(x_coarse, &size_coarse);
        
        // 限制：每个粗单元对应8个细单元（2x2x2）
        // 粗单元(i,j,k)对应细单元(2i:2i+1, 2j:2j+1, 2k:2k+1)
        // 注意：这里假设密度向量是按照(i,j,k)顺序存储的
        for (PetscInt k = 0; k < ne_coarse[2]; k++) {
            for (PetscInt j = 0; j < ne_coarse[1]; j++) {
                for (PetscInt i = 0; i < ne_coarse[0]; i++) {
                    PetscInt idx_coarse = i + j * ne_coarse[0] + k * ne_coarse[0] * ne_coarse[1];
                    
                    if (idx_coarse >= size_coarse) {
                        continue;
                    }
                    
                    // 对应的8个细单元
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
                                    
                                    if (idx_fine < size_fine) {
                                        sum += xf[idx_fine];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                    
                    xc[idx_coarse] = (count > 0) ? (sum / count) : 0.0;
                }
            }
        }
        
        VecRestoreArray(x_fine, &xf);
        VecRestoreArray(x_coarse, &xc);
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

// 组装指定层的粗网格刚度矩阵（几何重离散化）
PetscErrorCode LinearElasticity::AssembleStiffnessMatrix_Level(PetscInt level, DM da_level, Vec xPhys_level,
                                                                Mat K_level, Vec N_level,
                                                                PetscScalar Emin, PetscScalar Emax, PetscScalar penal) {
    PetscErrorCode ierr;
    
    // 获取单元信息
    PetscInt nel, nen;
    const PetscInt* necon;
    ierr = DMDAGetElements_3D(da_level, &nel, &nen, &necon);
    CHKERRQ(ierr);
    
    // 获取密度指针
    PetscScalar* xp;
    VecGetArray(xPhys_level, &xp);
    
    // 清零矩阵
    MatZeroEntries(K_level);
    
    // Edof数组
    PetscInt edof[24];
    PetscScalar ke[24 * 24];
    
    // 遍历单元
    for (PetscInt i = 0; i < nel; i++) {
        // 获取单元节点的自由度
        for (PetscInt j = 0; j < nen; j++) {
            for (PetscInt k = 0; k < 3; k++) {
                edof[j * 3 + k] = 3 * necon[i * nen + j] + k;
            }
        }
        
        // 使用SIMP插值
        PetscScalar dens = Emin + PetscPowScalar(xp[i], penal) * (Emax - Emin);
        for (PetscInt k = 0; k < 24 * 24; k++) {
            ke[k] = KE[k] * dens;
        }
        
        // 添加到稀疏矩阵
        ierr = MatSetValuesLocal(K_level, 24, edof, 24, edof, ke, ADD_VALUES);
        CHKERRQ(ierr);
    }
    
    MatAssemblyBegin(K_level, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(K_level, MAT_FINAL_ASSEMBLY);
    
    // ========== 关键修复：应用Dirichlet边界条件 ==========
    // 计算该层的固定DOF
    IS fixed_is;
    ierr = ComputeFixedDOFs_Level(da_level, &fixed_is);
    CHKERRQ(ierr);
    
    // 使用MatZeroRowsColumnsIS应用边界条件
    // 这会将固定DOF对应的行和列清零，并在对角线上设置1.0
    ierr = MatZeroRowsColumnsIS(K_level, fixed_is, 1.0, NULL, NULL);
    CHKERRQ(ierr);
    
    // 清理
    ISDestroy(&fixed_is);
    VecRestoreArray(xPhys_level, &xp);
    DMDARestoreElements(da_level, &nel, &nen, &necon);
    
    return ierr;
}

// 组装所有层的刚度矩阵（已废弃 - 现在使用回调函数）
PetscErrorCode LinearElasticity::AssembleAllLevels(Vec xPhys, PetscScalar Emin, PetscScalar Emax, PetscScalar penal) {
    PetscErrorCode ierr;
    
    // 步骤1：密度限制
    ierr = RestrictDensity(xPhys);
    CHKERRQ(ierr);
    
    // 步骤2和3已由回调函数自动完成
    // 不再需要手动组装和设置算子
    
    return ierr;
}

