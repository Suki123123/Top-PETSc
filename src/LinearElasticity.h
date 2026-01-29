#ifndef __LINEARELASTICITY__
#define __LINEARELASTICITY__

#include <fstream>
#include <iostream>
#include <math.h>
#include <petsc.h>
#include <petsc/private/dmdaimpl.h>

/*
 Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013
 Updated: June 2019, Niels Aage
 Copyright (C) 2013-2019,

 Disclaimer:
 The authors reserves all rights but does not guaranty that the code is
 free from errors. Furthermore, we shall not be liable in any event
 caused by the use of the program.
*/

// 前向声明回调函数
static PetscErrorCode ComputeMatrix_Level(KSP ksp, Mat J, Mat P, void* ctx);

class LinearElasticity {

    // 声明回调函数为友元
    friend PetscErrorCode ComputeMatrix_Level(KSP ksp, Mat J, Mat P, void* ctx);

  public:
    // Constructor
    LinearElasticity(DM da_nodes);

    // Destructor
    ~LinearElasticity();

    //  Compute objective and constraints and sensitivities at once: GOOD FOR
    //  SELF_ADJOINT PROBLEMS
    PetscErrorCode ComputeObjectiveConstraintsSensitivities(PetscScalar* fx, PetscScalar* gx, Vec dfdx, Vec dgdx,
                                                            Vec xPhys, PetscScalar Emin, PetscScalar Emax,
                                                            PetscScalar penal, PetscScalar volfrac);

    // Compute objective and constraints for the optimiation
    PetscErrorCode ComputeObjectiveConstraints(PetscScalar* fx, PetscScalar* gx, Vec xPhys, PetscScalar Emin,
                                               PetscScalar Emax, PetscScalar penal, PetscScalar volfrac);

    // Compute sensitivities
    PetscErrorCode ComputeSensitivities(Vec dfdx, Vec dgdx, Vec xPhys, PetscScalar Emin, PetscScalar Emax,
                                        PetscScalar penal,
                                        PetscScalar volfrac); // needs ....

    // Restart writer
    PetscErrorCode WriteRestartFiles();

    // Get pointer to the FE solution
    Vec GetStateField() { return (U); };

    // Get pointer to DMDA
    DM GetDM() { return (da_nodal); };

    // Logical mesh
    DM da_nodal; // Nodal mesh

  private:
    // Logical mesh
    PetscInt    nn[3]; // Number of nodes in each direction
    PetscInt    ne[3]; // Number of elements in each direction
    PetscScalar xc[6]; // Domain coordinates

    // Linear algebra
    Mat         K;           // Global stiffness matrix
    Vec         U;           // Displacement vector
    Vec         RHS;         // Load vector
    Vec         N;           // Dirichlet vector (used when imposing BCs)
    PetscScalar KE[24 * 24]; // Element stiffness matrix
    // Solver
    KSP         ksp; // Pointer to the KSP object i.e. the linear solver+prec
    PetscInt    nlvls;
    PetscScalar nu; // Possions ratio
    
    // 多层网格矩阵（用于几何重离散化GMG）
    Mat* coarse_K;      // 粗网格刚度矩阵数组 [nlvls]
    DM*  coarse_da;     // 粗网格DM数组 [nlvls]
    Vec* coarse_N;      // 粗网格Dirichlet向量数组 [nlvls]
    Vec* density_levels; // 各层密度向量数组 [nlvls]
    Mat* interpolation; // 插值算子数组 [nlvls-1]
    PetscBool use_geometric_mg;  // 是否使用几何重离散化
    
    // 材料参数（用于回调函数）
    PetscScalar current_Emin;
    PetscScalar current_Emax;
    PetscScalar current_penal;

    // Set up the FE mesh and data structures
    PetscErrorCode SetUpLoadAndBC(DM da_nodes);

    // Solve the FE problem
    PetscErrorCode SolveState(Vec xPhys, PetscScalar Emin, PetscScalar Emax, PetscScalar penal);

    // Assemble the stiffness matrix
    PetscErrorCode AssembleStiffnessMatrix(Vec xPhys, PetscScalar Emin, PetscScalar Emax, PetscScalar penal);
    
    // 组装粗网格刚度矩阵（几何重离散化）
    PetscErrorCode AssembleStiffnessMatrix_Level(PetscInt level, DM da_level, Vec xPhys_level, 
                                                  Mat K_level, Vec N_level, 
                                                  PetscScalar Emin, PetscScalar Emax, PetscScalar penal);
    
    // 密度限制：从细网格限制到所有粗网格
    PetscErrorCode RestrictDensity(Vec xPhys_fine);
    
    // 计算固定自由度（边界条件）
    PetscErrorCode ComputeFixedDOFs_Level(DM dm, IS* fixed_is);
    
    // 组装所有层的刚度矩阵
    PetscErrorCode AssembleAllLevels(Vec xPhys, PetscScalar Emin, PetscScalar Emax, PetscScalar penal);

    // Start the solver
    PetscErrorCode SetUpSolver();
    
    // 初始化多层网格（warm-up assembly）
    PetscErrorCode InitializeMultiGrid(Vec xPhys_initial, PetscScalar Emin, PetscScalar Emax, PetscScalar penal);

    // Routine that doesn't change the element type upon repeated calls
    PetscErrorCode DMDAGetElements_3D(DM dm, PetscInt* nel, PetscInt* nen, const PetscInt* e[]);

    // Methods used to assemble the element stiffness matrix
    PetscInt    Hex8Isoparametric(PetscScalar* X, PetscScalar* Y, PetscScalar* Z, PetscScalar nu, PetscInt redInt,
                                  PetscScalar* ke);
    PetscScalar Dot(PetscScalar* v1, PetscScalar* v2, PetscInt l);
    void        DifferentiatedShapeFunctions(PetscScalar xi, PetscScalar eta, PetscScalar zeta, PetscScalar* dNdxi,
                                             PetscScalar* dNdeta, PetscScalar* dNdzeta);
    PetscScalar Inverse3M(PetscScalar J[][3], PetscScalar invJ[][3]);

    // Restart
    PetscBool   restart, flip;
    std::string filename00, filename01;

    // File existence
    inline PetscBool fexists(const std::string& filename) {
        std::ifstream ifile(filename.c_str());
        if (ifile) {
            return PETSC_TRUE;
        }
        return PETSC_FALSE;
    }
};

#endif
