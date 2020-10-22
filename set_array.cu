/* Test for a simple CUDA kernel which simply sets an array to a constant value */
#include <petscsystypes.h>

__global__ void set_constant_value(PetscScalar *device_array, PetscInt n, PetscScalar value)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) device_array[i] = value;
}

extern "C"
PetscErrorCode set_device_array_constant(PetscScalar *device_array, PetscInt n, PetscScalar value)
{
  set_constant_value<<<(n+127)/128,128>>>(device_array, n, value);
  return 0;
}
