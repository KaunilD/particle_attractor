#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
__device__ float dot(float4 a, float4 b){
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__global__ void kernel_update(
	float4* d_positions, float4* d_velocities,
	float* d_masses, size_t numel) {

	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if (col >= numel) { return; }

	d_positions[col] = make_float4(
		min(1.0f, max(-1.0f, d_positions[col].x + d_velocities[col].x*5)),
		min(1.0f, max(-1.0f, d_positions[col].y + d_velocities[col].y*5)),
		min(1.0f, max(-1.0f, d_positions[col].z + d_velocities[col].z)),
		1.0f
	);

	__syncthreads();
}

void launch_kernel_update(
	float4* d_positions, float4* d_velocities, 
	float* masses, int numel) {
	dim3 blockSize(1024, 1, 1);
	dim3 gridSize(1);
	gridSize.x = numel / blockSize.x;
	kernel_update << <gridSize, blockSize >> > (
		d_positions, d_velocities,
		masses, numel
	);

	cudaDeviceSynchronize();
}