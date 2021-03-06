#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda/utils.hpp"

__device__ float dot(float4 a, float4 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__global__ void kernel_update(
	float4* d_positions, float4* d_og_positions, float4* d_velocities,
	float* d_masses, size_t numel) {

	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if (col >= numel) { return; }

	float4 velocity = d_velocities[col];

	float mag = sqrtf(velocity.x*velocity.x + velocity.y*velocity.y)*0.03;
	float pos = min(mag, 0.50f);
	d_positions[col] = make_float4(
		d_og_positions[col].x,
		d_og_positions[col].y,
		pos, 0
	);
	__syncthreads();
}

void launch_kernel_update(
	float4* d_positions, float4* d_og_positions, float4* d_velocities,
	float* masses, int numel) {
	
	dim3 blockSize = { 1024 };
	dim3 gridSize = {
		static_cast<unsigned int>((numel + blockSize.x - 1) / blockSize.x)
	};
	kernel_update << <gridSize, blockSize >> > (
		d_positions, d_og_positions, d_velocities,
		masses, numel
		);

	CUDACHECK(cudaDeviceSynchronize());
}