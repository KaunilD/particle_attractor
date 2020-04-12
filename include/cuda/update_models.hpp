#include <cuda.h>
#include <cuda_runtime.h>


void launch_kernel_models(
	float4* d_positions,
	float* d_modelBuffer,
	size_t num_bytes
);
