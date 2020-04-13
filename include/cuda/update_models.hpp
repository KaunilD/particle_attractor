#include <cuda.h>
#include <cuda_runtime.h>


void launch_kernel_models(
	float4* d_positions,
	float4* d_modelBuffer,
	int num_bytes
);
