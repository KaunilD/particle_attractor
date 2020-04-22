#include <cuda.h>
#include <cuda_runtime.h>
#include<iostream>

__global__ void kernel_update_velocities(float4* d_uv, float4* d_velocities_buffer, int numel) {

	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if (col >= numel) { return; }

	d_velocities_buffer[col] = make_float4(
		d_uv[col].x,
		d_uv[col].y,
		0,
		0
	);
	__syncthreads();
}


void launch_kernel_velocities(float4* d_uv, float4* d_velocities_buffer, int numel) {
	dim3 blockSize(1024, 1, 1);
	dim3 gridSize(1);
	gridSize.x = numel / blockSize.x + 1;

	kernel_update_velocities << < gridSize, blockSize >> > (d_uv, d_velocities_buffer, numel);

	cudaDeviceSynchronize();
}