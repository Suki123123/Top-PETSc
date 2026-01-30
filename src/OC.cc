/**
 * OC.cc - Optimality Criteria (OC) optimizer with GPU support
 */

#include "OC.h"
#include "MatrixFreeGPU.h"
#include <cmath>
#include <mpi.h>

OC::OC(PetscInt n_global, Vec x) {
    n = n_global;
    VecGetLocalSize(x, &n_local);
    move = 0.2;

    // Allocate workspace vectors
    VecDuplicate(x, &xold);
    VecDuplicate(x, &xnew);

    // Check if using GPU
    use_gpu = PETSC_FALSE;
    VecType vec_type;
    VecGetType(x, &vec_type);
    if (vec_type && (strcmp(vec_type, VECCUDA) == 0 ||
                     strcmp(vec_type, VECMPICUDA) == 0 ||
                     strcmp(vec_type, VECSEQCUDA) == 0)) {
        use_gpu = PETSC_TRUE;
    }
}

OC::~OC() {
    VecDestroy(&xold);
    VecDestroy(&xnew);
}

PetscErrorCode OC::SetOuterMovelimit(PetscScalar Xmin, PetscScalar Xmax,
                                     PetscScalar movlim, Vec x,
                                     Vec xmin_vec, Vec xmax_vec) {
    PetscErrorCode ierr;
    move = movlim;

    if (use_gpu) {
        // GPU path
        const PetscScalar* d_x;
        PetscScalar *d_xmin, *d_xmax;

        ierr = VecCUDAGetArrayRead(x, &d_x); CHKERRQ(ierr);
        ierr = VecCUDAGetArray(xmin_vec, &d_xmin); CHKERRQ(ierr);
        ierr = VecCUDAGetArray(xmax_vec, &d_xmax); CHKERRQ(ierr);

        OCGPU_SetOuterMovelimit(d_xmin, d_xmax, d_x, n_local, Xmin, Xmax, move);

        ierr = VecCUDARestoreArrayRead(x, &d_x); CHKERRQ(ierr);
        ierr = VecCUDARestoreArray(xmin_vec, &d_xmin); CHKERRQ(ierr);
        ierr = VecCUDARestoreArray(xmax_vec, &d_xmax); CHKERRQ(ierr);
    } else {
        // CPU path
        PetscScalar *xp, *xminp, *xmaxp;
        VecGetArray(x, &xp);
        VecGetArray(xmin_vec, &xminp);
        VecGetArray(xmax_vec, &xmaxp);

        for (PetscInt i = 0; i < n_local; i++) {
            xminp[i] = PetscMax(Xmin, xp[i] - move);
            xmaxp[i] = PetscMin(Xmax, xp[i] + move);
        }

        VecRestoreArray(x, &xp);
        VecRestoreArray(xmin_vec, &xminp);
        VecRestoreArray(xmax_vec, &xmaxp);
    }

    return 0;
}

PetscErrorCode OC::Update(Vec x, Vec dfdx, PetscScalar* gx, Vec* dgdx,
                          Vec xmin_vec, Vec xmax_vec) {
    PetscErrorCode ierr;

    // Save old design
    VecCopy(x, xold);

    if (use_gpu) {
        // GPU path using OCGPU functions
        PetscScalar *d_x, *d_xold, *d_xnew;
        const PetscScalar *d_dfdx, *d_dgdx;

        ierr = VecCUDAGetArray(x, &d_x); CHKERRQ(ierr);
        ierr = VecCUDAGetArray(xold, &d_xold); CHKERRQ(ierr);
        ierr = VecCUDAGetArray(xnew, &d_xnew); CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(dfdx, &d_dfdx); CHKERRQ(ierr);
        ierr = VecCUDAGetArrayRead(dgdx[0], &d_dgdx); CHKERRQ(ierr);

        // Copy x to xold on GPU
        OCGPU_VecCopy(d_xold, d_x, n_local);

        // Bisection to find lambda
        PetscScalar l1 = 1e-9;
        PetscScalar l2 = 1e9;
        PetscScalar lmid;
        PetscScalar vol_local, vol_global;
        PetscScalar target_vol = gx[0] + 1.0;  // gx[0] = (sum(x)/n - volfrac), so target = volfrac * n

        // Get volfrac from constraint: gx[0] = sum(x)/n - volfrac
        // We need to find target volume = volfrac * n
        // From gx[0] = sum(x)/n - volfrac, we have volfrac = sum(x)/n - gx[0]
        // But we want to satisfy gx[0] <= 0, so sum(x)/n <= volfrac
        // Target volume = volfrac * n

        // Get current volume
        PetscScalar current_vol_local;
        OCGPU_ComputeVolume(&current_vol_local, d_x, n_local);
        PetscScalar current_vol_global;
        MPI_Allreduce(&current_vol_local, &current_vol_global, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

        // volfrac = current_vol / n - gx[0]
        PetscScalar volfrac = current_vol_global / n - gx[0];
        target_vol = volfrac * n;

        PetscInt max_iter = 100;
        for (PetscInt iter = 0; iter < max_iter; iter++) {
            lmid = 0.5 * (l1 + l2);

            // Update design with current lambda
            OCGPU_Update(d_xnew, d_xold, d_dfdx, d_dgdx, n_local,
                        lmid, move, 0.0, 1.0);

            // Compute volume
            OCGPU_ComputeVolume(&vol_local, d_xnew, n_local);
            MPI_Allreduce(&vol_local, &vol_global, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

            // Update bisection bounds
            if (vol_global > target_vol) {
                l1 = lmid;
            } else {
                l2 = lmid;
            }

            // Check convergence
            if ((l2 - l1) / (l1 + l2) < 1e-4) {
                break;
            }
        }

        // Copy result to x
        OCGPU_VecCopy(d_x, d_xnew, n_local);

        ierr = VecCUDARestoreArray(x, &d_x); CHKERRQ(ierr);
        ierr = VecCUDARestoreArray(xold, &d_xold); CHKERRQ(ierr);
        ierr = VecCUDARestoreArray(xnew, &d_xnew); CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(dfdx, &d_dfdx); CHKERRQ(ierr);
        ierr = VecCUDARestoreArrayRead(dgdx[0], &d_dgdx); CHKERRQ(ierr);

    } else {
        // CPU path
        PetscScalar *xp, *xoldp, *xnewp, *dfdxp, *dgdxp;
        VecGetArray(x, &xp);
        VecGetArray(xold, &xoldp);
        VecGetArray(xnew, &xnewp);
        VecGetArray(dfdx, &dfdxp);
        VecGetArray(dgdx[0], &dgdxp);

        // Copy x to xold
        for (PetscInt i = 0; i < n_local; i++) {
            xoldp[i] = xp[i];
        }

        // Get current volume and compute target
        PetscScalar vol_local = 0.0;
        for (PetscInt i = 0; i < n_local; i++) {
            vol_local += xp[i];
        }
        PetscScalar vol_global;
        MPI_Allreduce(&vol_local, &vol_global, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

        PetscScalar volfrac = vol_global / n - gx[0];
        PetscScalar target_vol = volfrac * n;

        // Bisection to find lambda
        PetscScalar l1 = 1e-9;
        PetscScalar l2 = 1e9;
        PetscScalar lmid;

        PetscInt max_iter = 100;
        for (PetscInt iter = 0; iter < max_iter; iter++) {
            lmid = 0.5 * (l1 + l2);

            // Update design with current lambda
            vol_local = 0.0;
            for (PetscInt i = 0; i < n_local; i++) {
                PetscScalar Be = -dfdxp[i] / (lmid * dgdxp[i]);
                if (Be < 0.0) Be = 0.0;
                PetscScalar x_oc = xoldp[i] * sqrt(Be);

                // Apply move limits
                PetscScalar x_lower = PetscMax(0.0, xoldp[i] - move);
                PetscScalar x_upper = PetscMin(1.0, xoldp[i] + move);
                xnewp[i] = PetscMax(x_lower, PetscMin(x_upper, x_oc));

                vol_local += xnewp[i];
            }

            // Compute global volume
            MPI_Allreduce(&vol_local, &vol_global, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

            // Update bisection bounds
            if (vol_global > target_vol) {
                l1 = lmid;
            } else {
                l2 = lmid;
            }

            // Check convergence
            if ((l2 - l1) / (l1 + l2) < 1e-4) {
                break;
            }
        }

        // Copy result to x
        for (PetscInt i = 0; i < n_local; i++) {
            xp[i] = xnewp[i];
        }

        VecRestoreArray(x, &xp);
        VecRestoreArray(xold, &xoldp);
        VecRestoreArray(xnew, &xnewp);
        VecRestoreArray(dfdx, &dfdxp);
        VecRestoreArray(dgdx[0], &dgdxp);
    }

    return 0;
}

PetscScalar OC::DesignChange(Vec x, Vec xold_ext) {
    PetscScalar ch_local = 0.0;
    PetscScalar ch_global;

    if (use_gpu) {
        const PetscScalar *d_x, *d_xold;
        VecCUDAGetArrayRead(x, &d_x);
        VecCUDAGetArrayRead(xold, &d_xold);
        OCGPU_ComputeChange(&ch_local, d_x, d_xold, n_local);
        VecCUDARestoreArrayRead(x, &d_x);
        VecCUDARestoreArrayRead(xold, &d_xold);
    } else {
        PetscScalar *xp, *xoldp;
        VecGetArray(x, &xp);
        VecGetArray(xold, &xoldp);

        for (PetscInt i = 0; i < n_local; i++) {
            PetscScalar diff = PetscAbsScalar(xp[i] - xoldp[i]);
            if (diff > ch_local) ch_local = diff;
        }

        VecRestoreArray(x, &xp);
        VecRestoreArray(xold, &xoldp);
    }

    MPI_Allreduce(&ch_local, &ch_global, 1, MPIU_SCALAR, MPI_MAX, PETSC_COMM_WORLD);

    // Also update xold_ext for external use
    VecCopy(xold, xold_ext);

    return ch_global;
}
