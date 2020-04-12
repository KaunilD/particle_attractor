#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_update_models(float4* d_positions, float* d_modelBuffer, size_t num_bytes) {

	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if (col >= num_bytes/64) { return; }

	//float4 position = d_positions[col];
	
	d_modelBuffer[col] = d_modelBuffer[col] + d_positions[col].x;
	d_modelBuffer[col+1] = d_modelBuffer[col+1] + d_positions[col].y;
	d_modelBuffer[col+2] = d_modelBuffer[col+2] + d_positions[col].z;
}

void launch_kernel_models(float4* d_positions, float* d_modelBuffer, size_t num_bytes) {
	dim3 blockSize(32, 1, 1);
	dim3 gridSize(1);
	gridSize.x = (num_bytes / 64)/blockSize.x;
	kernel_update_models << <gridSize, blockSize >> > (
		d_positions, d_modelBuffer, num_bytes
	);

	cudaDeviceSynchronize();
}