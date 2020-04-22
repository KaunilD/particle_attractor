#include <cuda.h>
#include <cuda_runtime.h>


void launch_kernel_velocities(
	float4* d_uv,
	float4* d_velocitiesBuffer,
	int num_bytes
);
