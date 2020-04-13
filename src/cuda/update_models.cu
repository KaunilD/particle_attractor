#include <cuda.h>
#include <cuda_runtime.h>
#include<iostream>

__global__ void kernel_update_models(float4* d_positions, float4* d_modelBuffer, int numel) {

	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if (col >= numel) { return; }

	d_modelBuffer[col*4+3] = make_float4(
		d_positions[col].x,
		d_positions[col].y,
		d_positions[col].z,
		1
	);

	/*
	if (col%32 == 0) {

		for (int i = 0; i < 4; i++) {
			printf("%f, %f, %f, %f\n", d_modelBuffer[col + i].x, d_modelBuffer[col + i].y, d_modelBuffer[col + i].z, d_modelBuffer[col + i].w);
		}
		printf("\n");
	}
	*/
	__syncthreads();
}

void launch_kernel_models(float4* d_positions, float4* d_modelBuffer, int numel) {
	dim3 blockSize(1024, 1, 1);
	dim3 gridSize(1);
	gridSize.x = numel / blockSize.x;

	kernel_update_models <<< gridSize, blockSize >>> (d_positions, d_modelBuffer, numel);

	cudaDeviceSynchronize();
}