/**
 * OC.h - Optimality Criteria (OC) optimizer with GPU support
 */

#ifndef OC_H
#define OC_H

#include <petsc.h>

class OC {
public:
    OC(PetscInt n, Vec x);
    ~OC();

    // Update design variables using OC method
    PetscErrorCode Update(Vec x, Vec dfdx, PetscScalar* gx, Vec* dgdx,
                          Vec xmin, Vec xmax);

    // Set move limit
    PetscErrorCode SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax,
                                     PetscScalar movlim, Vec x,
                                     Vec xmin, Vec xmax);

    // Compute design change (infinity norm)
    PetscScalar DesignChange(Vec x, Vec xold);

private:
    PetscInt n;           // Number of design variables (global)
    PetscInt n_local;     // Number of local design variables
    PetscScalar move;     // Move limit
    Vec xold;             // Old design
    Vec xnew;             // New design (workspace)

    // GPU support
    PetscBool use_gpu;
};

#endif // OC_H
