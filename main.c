static char help[] = "Steal a device pointer from a PETSc CUDA vector and call custom CUDA on it\n\n";

#include <petsc.h>

PetscErrorCode set_device_array_constant(PetscScalar*,PetscInt,PetscScalar);

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  DM             dm;
  Vec            vec;
  PetscScalar    *device_array;
  PetscInt       entries_local;

  ierr = PetscInitialize(&argc, &args, (char*)0, help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Not implemented for >1 rank");

  /* Create a Stokes-style DMStag */
  ierr = DMStagCreate2d(
      PETSC_COMM_WORLD,
      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
      3,2,  // element counts
      PETSC_DECIDE, PETSC_DECIDE,
      0, 1, 1,
      DMSTAG_STENCIL_BOX, 1,
      NULL, NULL,
      &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);

  if (!PetscDefined(HAVE_CUDA)) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"PETSc must be configured with CUDA");

  /* Create a Vec backed by CUDA device memory */
  ierr = DMSetVecType(dm, VECCUDA);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &vec);CHKERRQ(ierr);

  /* Steal the pointer and apply a custom CUDA operation */
  ierr = VecCUDAGetArray(vec, &device_array);CHKERRQ(ierr);

  /* Test function with a simple CUDA kernel */
  ierr = DMStagGetEntriesLocal(dm, &entries_local);CHKERRQ(ierr);
  ierr = set_device_array_constant(device_array, entries_local, 3.2345);CHKERRQ(ierr);
  ierr = VecCUDARestoreArray(vec, &device_array);CHKERRQ(ierr);
  ierr = VecView(vec,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); // should print constant values
  ierr = DMRestoreLocalVector(dm, &vec);CHKERRQ(ierr);

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
