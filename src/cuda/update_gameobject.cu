#include <cuda.h>
#include <cuda_runtime.h>

__device__ float dot(float4 a, float4 b){
	return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

__global__ void kernel_update(
	float4* d_positions, float4* d_velocities,
	float4* h_positions, float4* h_velocities,
	float* d_masses, size_t numel) {

	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if (col >= numel) { return; }

	float4 distance = d_positions[col];
	float magnitude = sqrt(
		distance.x * distance.x +
		distance.y * distance.y +
		distance.z * distance.z
	);
	magnitude			= min(30.0f, max(magnitude, 5.0f));
	float force			= (0.4 * d_masses[0]) / (magnitude);
	float coeff			= rsqrt(dot(distance, distance))*force;

	float4 acceleration	= make_float4(
		coeff*-distance.x, 
		coeff*-distance.y, 
		coeff*-distance.z, 
		coeff*-distance.w
	);	
	
	
	d_velocities[col]	= make_float4(
		d_velocities[col].x +acceleration.x * 0.05f,
		d_velocities[col].y +acceleration.y * 0.05f,
		d_velocities[col].z +acceleration.z * 0.05f,
		d_velocities[col].w +acceleration.w * 0.05f
	);

	d_positions[col]	= make_float4(
		d_positions[col].x + d_velocities[col].x * 0.05f,
		d_positions[col].y + d_velocities[col].y * 0.05f,
		d_positions[col].z + d_velocities[col].z * 0.05f,
		d_positions[col].w + d_velocities[col].w * 0.05f
	);

}

void launch_kernel_update(
	float4* d_positions, float4* d_velocities, 
	float4* h_positions, float4* h_velocities, 
	float* masses, int numel) {
	dim3 blockSize(32, 1, 1);
	dim3 gridSize(1);
	gridSize.x = numel / blockSize.x;
	kernel_update << <gridSize, blockSize >> > (
		d_positions, d_velocities,
		h_positions, h_velocities,
		masses, numel
	);

	cudaDeviceSynchronize();
}