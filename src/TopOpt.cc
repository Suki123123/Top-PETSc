#include "TopOpt.h"
#include "MMA.h"
#include <cmath>

/*
 Authors: Niels Aage, Erik Andreassen, Boyan Lazarov, August 2013
 Updated: June 2019, Niels Aage
 Copyright (C) 2013-2019,

 Disclaimer:
 The authors reserves all rights but does not guaranty that the code is
 free from errors. Furthermore, we shall not be liable in any event
 caused by the use of the program.
*/

TopOpt::TopOpt(PetscInt nconstraints) {

    m = nconstraints;
    Init();
}

TopOpt::TopOpt() {

    m = 1;
    Init();
}

void TopOpt::Init() { // Dummy constructor

    x        = NULL;
    xPhys    = NULL;
    dfdx     = NULL;
    dgdx     = NULL;
    gx       = NULL;
    da_nodes = NULL;
    da_elem  = NULL;

    SetUp();
}

TopOpt::~TopOpt() {

    // Delete vectors
    if (x != NULL) {
        VecDestroy(&x);
    }
    if (xTilde != NULL) {
        VecDestroy(&xTilde);
    }
    if (xPhys != NULL) {
        VecDestroy(&xPhys);
    }
    if (dfdx != NULL) {
        VecDestroy(&dfdx);
    }
    if (dgdx != NULL) {
        VecDestroyVecs(m, &dgdx);
    }
    if (xold != NULL) {
        VecDestroy(&xold);
    }
    if (xmin != NULL) {
        VecDestroy(&xmin);
    }
    if (xmax != NULL) {
        VecDestroy(&xmax);
    }

    if (da_nodes != NULL) {
        DMDestroy(&(da_nodes));
    }
    if (da_elem != NULL) {
        DMDestroy(&(da_elem));
    }

    // Delete constraints
    if (gx != NULL) {
        delete[] gx;
    }
}

// NO METHODS !
// PetscErrorCode TopOpt::SetUp(Vec CRAPPY_VEC){
PetscErrorCode TopOpt::SetUp() {
    PetscErrorCode ierr;

    // SET DEFAULTS for FE mesh and levels for MG solver
    nxyz[0] = 65; // 129;
    nxyz[1] = 33; // 65;
    nxyz[2] = 33; // 65;
    xc[0]   = 0.0;
    xc[1]   = 2.0;
    xc[2]   = 0.0;
    xc[3]   = 1.0;
    xc[4]   = 0.0;
    xc[5]   = 1.0;
    nu      = 0.3;
    nlvls   = 1;  // PCG模式：不使用多重网格

    // SET DEFAULTS for optimization problems
    volfrac = 0.12;
    maxItr  = 200;  // 默认200次迭代
    rmin    = -1.0;  // -1表示自动计算（1.5倍单元尺寸）
    penal   = 3.0;
    Emin    = 1.0e-9;
    Emax    = 1.0;
    filter  = 2;  // 固定使用PDE滤波器
    Xmin    = 0.0;
    Xmax    = 1.0;
    movlim  = 0.2;

    // Projection filter
    projectionFilter = PETSC_FALSE;
    beta             = 0.1;
    betaFinal        = 48;
    eta              = 0.0;

    ierr = SetUpMESH();
    CHKERRQ(ierr);

    // 自适应计算rmin（如果未指定）
    // rmin = 1.5 * dx，保证滤波半径约为1.5个单元
    if (rmin < 0) {
        rmin = 1.5 * dx;
    }

    ierr = SetUpOPT();
    CHKERRQ(ierr);

    return (ierr);
}

PetscErrorCode TopOpt::SetUpMESH() {

    PetscErrorCode ierr;

    // Read input from arguments
    PetscBool flg;

    // 只保留网格参数（必需）
    PetscOptionsGetInt(NULL, NULL, "-nx", &(nxyz[0]), &flg);
    PetscOptionsGetInt(NULL, NULL, "-ny", &(nxyz[1]), &flg);
    PetscOptionsGetInt(NULL, NULL, "-nz", &(nxyz[2]), &flg);
    
    // 其他参数都使用默认值（已在Init()中设置）

    // Write parameters for the physics _ OWNED BY TOPOPT
    PetscPrintf(PETSC_COMM_WORLD, "##############################################"
                                  "##########################\n");
    PetscPrintf(PETSC_COMM_WORLD, "############################ FEM settings "
                                  "##############################\n");
    PetscPrintf(PETSC_COMM_WORLD, "# Number of nodes: (-nx,-ny,-nz):        (%i,%i,%i) \n", nxyz[0], nxyz[1], nxyz[2]);
    PetscPrintf(PETSC_COMM_WORLD, "# Number of degree of freedom:           %i \n", 3 * nxyz[0] * nxyz[1] * nxyz[2]);
    PetscPrintf(PETSC_COMM_WORLD, "# Number of elements:                    (%i,%i,%i) \n", nxyz[0] - 1, nxyz[1] - 1,
                nxyz[2] - 1);
    PetscPrintf(PETSC_COMM_WORLD, "# Dimensions: (-xcmin,-xcmax,..,-zcmax): (%f,%f,%f)\n", xc[1] - xc[0], xc[3] - xc[2],
                xc[5] - xc[4]);
    PetscPrintf(PETSC_COMM_WORLD, "# -nlvls: %i\n", nlvls);
    PetscPrintf(PETSC_COMM_WORLD, "##############################################"
                                  "##########################\n");

    // Check if the mesh supports the chosen number of MG levels
    PetscScalar divisor = PetscPowScalar(2.0, (PetscScalar)nlvls - 1.0);
    // x - dir
    if (std::floor((PetscScalar)(nxyz[0] - 1) / divisor) != (nxyz[0] - 1.0) / ((PetscInt)divisor)) {
        PetscPrintf(PETSC_COMM_WORLD, "MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD, "X - number of nodes %i is cannot be halfened %i times\n", nxyz[0], nlvls - 1);
        exit(0);
    }
    // y - dir
    if (std::floor((PetscScalar)(nxyz[1] - 1) / divisor) != (nxyz[1] - 1.0) / ((PetscInt)divisor)) {
        PetscPrintf(PETSC_COMM_WORLD, "MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD, "Y - number of nodes %i is cannot be halfened %i times\n", nxyz[1], nlvls - 1);
        exit(0);
    }
    // z - dir
    if (std::floor((PetscScalar)(nxyz[2] - 1) / divisor) != (nxyz[2] - 1.0) / ((PetscInt)divisor)) {
        PetscPrintf(PETSC_COMM_WORLD, "MESH DIMENSION NOT COMPATIBLE WITH NUMBER OF MULTIGRID LEVELS!\n");
        PetscPrintf(PETSC_COMM_WORLD, "Z - number of nodes %i is cannot be halfened %i times\n", nxyz[2], nlvls - 1);
        exit(0);
    }

    // Start setting up the FE problem
    // Boundary types: DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_GHOSTED,
    // DMDA_BOUNDARY_PERIODIC
    DMBoundaryType bx = DM_BOUNDARY_NONE;
    DMBoundaryType by = DM_BOUNDARY_NONE;
    DMBoundaryType bz = DM_BOUNDARY_NONE;

    // Stencil type - box since this is closest to FEM (i.e. STAR is FV/FD)
    DMDAStencilType stype = DMDA_STENCIL_BOX;

    // Discretization: nodes:
    // For standard FE - number must be odd
    // FOr periodic: Number must be even
    PetscInt nx = nxyz[0];
    PetscInt ny = nxyz[1];
    PetscInt nz = nxyz[2];

    // number of nodal dofs: Nodal design variable - NOT REALLY NEEDED
    PetscInt numnodaldof = 1;

    // Stencil width: each node connects to a box around it - linear elements
    PetscInt stencilwidth = 1;

    // Coordinates and element sizes: note that dx,dy,dz are half the element size
    PetscReal xmin = xc[0], xmax = xc[1], ymin = xc[2], ymax = xc[3], zmin = xc[4], zmax = xc[5];
    dx = (xc[1] - xc[0]) / (PetscScalar(nxyz[0] - 1));
    dy = (xc[3] - xc[2]) / (PetscScalar(nxyz[1] - 1));
    dz = (xc[5] - xc[4]) / (PetscScalar(nxyz[2] - 1));

    // Create the nodal mesh
    ierr = DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, stype, nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                        numnodaldof, stencilwidth, 0, 0, 0, &(da_nodes));
    CHKERRQ(ierr);

    // Initialize
    DMSetFromOptions(da_nodes);
    DMSetUp(da_nodes);

    // Set the coordinates
    ierr = DMDASetUniformCoordinates(da_nodes, xmin, xmax, ymin, ymax, zmin, zmax);
    CHKERRQ(ierr);

    // Set the element type to Q1: Otherwise calls to GetElements will change to
    // P1 ! STILL DOESN*T WORK !!!!
    ierr = DMDASetElementType(da_nodes, DMDA_ELEMENT_Q1);
    CHKERRQ(ierr);

    // Create the element mesh: NOTE THIS DOES NOT INCLUDE THE FILTER !!!
    // find the geometric partitioning of the nodal mesh, so the element mesh will
    // coincide with the nodal mesh
    PetscInt md, nd, pd;
    ierr = DMDAGetInfo(da_nodes, NULL, NULL, NULL, NULL, &md, &nd, &pd, NULL, NULL, NULL, NULL, NULL, NULL);
    CHKERRQ(ierr);

    // vectors with partitioning information
    PetscInt* Lx = new PetscInt[md];
    PetscInt* Ly = new PetscInt[nd];
    PetscInt* Lz = new PetscInt[pd];

    // get number of nodes for each partition
    const PetscInt *LxCorrect, *LyCorrect, *LzCorrect;
    ierr = DMDAGetOwnershipRanges(da_nodes, &LxCorrect, &LyCorrect, &LzCorrect);
    CHKERRQ(ierr);

    // subtract one from the lower left corner.
    for (int i = 0; i < md; i++) {
        Lx[i] = LxCorrect[i];
        if (i == 0) {
            Lx[i] = Lx[i] - 1;
        }
    }
    for (int i = 0; i < nd; i++) {
        Ly[i] = LyCorrect[i];
        if (i == 0) {
            Ly[i] = Ly[i] - 1;
        }
    }
    for (int i = 0; i < pd; i++) {
        Lz[i] = LzCorrect[i];
        if (i == 0) {
            Lz[i] = Lz[i] - 1;
        }
    }

    // Create the element grid: NOTE CONNECTIVITY IS 0
    PetscInt conn = 0;
    ierr = DMDACreate3d(PETSC_COMM_WORLD, bx, by, bz, stype, nx - 1, ny - 1, nz - 1, md, nd, pd, 1, conn, Lx, Ly, Lz,
                        &(da_elem));
    CHKERRQ(ierr);

    // Initialize
    DMSetFromOptions(da_elem);
    DMSetUp(da_elem);

    // Set element center coordinates
    ierr = DMDASetUniformCoordinates(da_elem, xmin + dx / 2.0, xmax - dx / 2.0, ymin + dy / 2.0, ymax - dy / 2.0,
                                     zmin + dz / 2.0, zmax - dz / 2.0);
    CHKERRQ(ierr);

    // Clean up
    delete[] Lx;
    delete[] Ly;
    delete[] Lz;

    return (ierr);
}

PetscErrorCode TopOpt::SetUpOPT() {

    PetscErrorCode ierr;

    // ierr = VecDuplicate(CRAPPY_VEC,&xPhys); CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da_elem, &xPhys);
    CHKERRQ(ierr);
    // Total number of design variables
    VecGetSize(xPhys, &n);

    PetscBool flg;

    // 只保留最大迭代次数参数（可选，默认200）
    PetscOptionsGetInt(NULL, NULL, "-maxItr", &maxItr, &flg);
    
    // 其他所有参数都使用默认值（已在Init()中设置）

    PetscPrintf(PETSC_COMM_WORLD, "################### Optimization settings ####################\n");
    PetscPrintf(PETSC_COMM_WORLD, "# Problem size: n= %i, m= %i\n", n, m);
    PetscPrintf(PETSC_COMM_WORLD, "# Filter: PDE (rmin: %f, %.1f elements)\n", rmin, rmin / dx);
    PetscPrintf(PETSC_COMM_WORLD, "# Volume fraction: %f\n", volfrac);
    PetscPrintf(PETSC_COMM_WORLD, "# Penalty factor: %f (fixed)\n", penal);
    PetscPrintf(PETSC_COMM_WORLD, "# Emin/Emax: %e / %e\n", Emin, Emax);
    PetscPrintf(PETSC_COMM_WORLD, "# Poisson's ratio: %f\n", nu);
    PetscPrintf(PETSC_COMM_WORLD, "# Max iterations: %i\n", maxItr);
    PetscPrintf(PETSC_COMM_WORLD, "# Move limit: %f\n", movlim);
    PetscPrintf(PETSC_COMM_WORLD, "# Convergence criterion: ch < 0.02\n");
    PetscPrintf(PETSC_COMM_WORLD, "##############################################################\n");

    // Allocate after input
    gx = new PetscScalar[m];
    if (filter == 0) {
        Xmin = 0.001; // Prevent division by zero in filter
    }

    // Allocate the optimization vectors
    ierr = VecDuplicate(xPhys, &x);
    CHKERRQ(ierr);
    ierr = VecDuplicate(xPhys, &xTilde);
    CHKERRQ(ierr);

    VecSet(x, volfrac);      // Initialize to volfrac !
    VecSet(xTilde, volfrac); // Initialize to volfrac !
    VecSet(xPhys, volfrac);  // Initialize to volfrac !

    // Sensitivity vectors
    ierr = VecDuplicate(x, &dfdx);
    CHKERRQ(ierr);
    ierr = VecDuplicateVecs(x, m, &dgdx);
    CHKERRQ(ierr);

    // Bounds and
    VecDuplicate(x, &xmin);
    VecDuplicate(x, &xmax);
    VecDuplicate(x, &xold);
    VecSet(xold, volfrac);

    return (ierr);
}

PetscErrorCode TopOpt::AllocateMMAwithRestart(PetscInt* itr, MMA** mma) {
    PetscErrorCode ierr = 0;

    // 检查是否存在restart文件
    std::string filename00 = "Restart00_xPhys.dat";
    std::string filename01 = "Restart01_xPhys.dat";

    // 如果没有restart文件，从头开始
    if (!fexists(filename00) || !fexists(filename01)) {
        PetscPrintf(PETSC_COMM_WORLD, "# 没有找到restart文件，从头开始优化\n");
        *itr = 0;
        *mma = new MMA(n, m, x);
        return ierr;
    }

    // 读取restart文件
    PetscPrintf(PETSC_COMM_WORLD, "# 从restart文件恢复优化\n");

    Vec xo1, xo2, U, L;
    VecDuplicate(x, &xo1);
    VecDuplicate(x, &xo2);
    VecDuplicate(x, &U);
    VecDuplicate(x, &L);

    // 读取xo1 (Restart00)
    PetscViewer view;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename00.c_str(), FILE_MODE_READ, &view);
    CHKERRQ(ierr);
    VecLoad(xo1, view);
    PetscViewerDestroy(&view);

    // 读取xo2 (Restart01)
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename01.c_str(), FILE_MODE_READ, &view);
    CHKERRQ(ierr);
    VecLoad(xo2, view);
    PetscViewerDestroy(&view);

    // 读取U
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "RestartU.dat", FILE_MODE_READ, &view);
    CHKERRQ(ierr);
    VecLoad(U, view);
    PetscViewerDestroy(&view);

    // 读取L
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "RestartL.dat", FILE_MODE_READ, &view);
    CHKERRQ(ierr);
    VecLoad(L, view);
    PetscViewerDestroy(&view);

    // 设置迭代次数（假设从文件名推断，这里简化为3）
    *itr = 3;

    // 创建MMA对象
    *mma = new MMA(n, m, *itr, xo1, xo2, U, L);

    // 清理临时向量
    VecDestroy(&xo1);
    VecDestroy(&xo2);
    VecDestroy(&U);
    VecDestroy(&L);

    return ierr;
}

PetscErrorCode TopOpt::WriteRestartFiles(PetscInt* itr, MMA* mma) {
    PetscErrorCode ierr = 0;

    if (mma == NULL) {
        return ierr;
    }

    // 获取restart数据
    Vec xo1, xo2, U, L;
    VecDuplicate(x, &xo1);
    VecDuplicate(x, &xo2);
    VecDuplicate(x, &U);
    VecDuplicate(x, &L);

    mma->Restart(xo1, xo2, U, L);

    // 写入文件
    PetscViewer view;
    
    // 写入xo1
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "Restart00_xPhys.dat", FILE_MODE_WRITE, &view);
    CHKERRQ(ierr);
    VecView(xo1, view);
    PetscViewerDestroy(&view);

    // 写入xo2
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "Restart01_xPhys.dat", FILE_MODE_WRITE, &view);
    CHKERRQ(ierr);
    VecView(xo2, view);
    PetscViewerDestroy(&view);

    // 写入U
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "RestartU.dat", FILE_MODE_WRITE, &view);
    CHKERRQ(ierr);
    VecView(U, view);
    PetscViewerDestroy(&view);

    // 写入L
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, "RestartL.dat", FILE_MODE_WRITE, &view);
    CHKERRQ(ierr);
    VecView(L, view);
    PetscViewerDestroy(&view);

    // 清理
    VecDestroy(&xo1);
    VecDestroy(&xo2);
    VecDestroy(&U);
    VecDestroy(&L);

    return ierr;
}
