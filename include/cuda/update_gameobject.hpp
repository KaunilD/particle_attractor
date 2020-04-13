#include <cuda.h>
#include <cuda_runtime.h>

void launch_kernel_update(float4* d_positions, float4* d_velocities, float4* h_positions, float4* h_velocities, float* masses, int numel);