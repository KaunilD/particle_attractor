#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
__device__ inline size_t rcToidx(size_t row, size_t col, size_t H, size_t W) {

	return row * W + col;
}

__global__ void kernel_partials(
	uchar3* d_f1ptr,
	float* d_f1dx, float* d_f1dy,
	int H, int W
) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = rcToidx(row, col, H, W);
	if (idx >= H * W || idx == 0) {
		return;
	}

	float gray_x1 = 0.2989f * (d_f1ptr[rcToidx(row, col - 1, H, W)].x/255.0f) + 
					0.5870f * (d_f1ptr[rcToidx(row, col - 1, H, W)].y/255.0f) + 
					0.1140f * (d_f1ptr[rcToidx(row, col - 1, H, W)].z/255.0f);

	float gray_x2 = 0.2989f * (d_f1ptr[rcToidx(row, col + 1, H, W)].x/255.0f) + 
					0.5870f * (d_f1ptr[rcToidx(row, col + 1, H, W)].y/255.0f) + 
					0.1140f * (d_f1ptr[rcToidx(row, col + 1, H, W)].z/255.0f);

	float gray_y1 = 0.2989f * (d_f1ptr[rcToidx(row - 1, col, H, W)].x/255.0f) + 
					0.5870f * (d_f1ptr[rcToidx(row - 1, col, H, W)].y/255.0f) + 
					0.1140f * (d_f1ptr[rcToidx(row - 1, col, H, W)].z/255.0f);

	float gray_y2 = 0.2989f * (d_f1ptr[rcToidx(row + 1, col, H, W)].x/255.0f) + 
					0.5870f * (d_f1ptr[rcToidx(row + 1, col, H, W)].y/255.0f) + 
					0.1140f * (d_f1ptr[rcToidx(row + 1, col, H, W)].z/255.0f);

	
	
	d_f1dx[idx] = (gray_x1 - gray_x2)/2.0f;
	d_f1dy[idx] = (gray_y1 - gray_y2)/2.0f;


}


void launch_partials(
	uchar3* d_f1ptr, 
	float* d_f1dx, float* d_f1dy, 
	int H, int W
) {

	const dim3 blockSize(32, 32);
	const dim3 gridSize(H / blockSize.x + 1, W / blockSize.y + 1);

	kernel_partials << <gridSize, blockSize >> > (
		d_f1ptr, d_f1dx, d_f1dy, H, W
	);

	cudaDeviceSynchronize();
}


__global__ void kernel_sub(uchar3* d_f1ptr, uchar3* d_f2ptr, float* d_dt, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = rcToidx(row, col, H, W);

	if (idx >= H * W || idx == 0) {
		return;
	}

	float gray_1 =	0.2989f * (d_f1ptr[idx].x / 255.0f) +
					0.5870f * (d_f1ptr[idx].y / 255.0f) +
					0.1140f * (d_f1ptr[idx].z / 255.0f);

	float gray_2 =	0.2989f * (d_f2ptr[idx].x / 255.0f) +
					0.5870f * (d_f2ptr[idx].y / 255.0f) +
					0.1140f * (d_f2ptr[idx].z / 255.0f);


	d_dt[idx] = gray_2 - gray_1;

}


void launch_sub(uchar3* d_f1ptr, uchar3* d_f2ptr, float* d_dt, int H, int W) {

	const dim3 blockSize(32, 32);
	const dim3 gridSize(H / blockSize.x + 1, W / blockSize.y + 1);

	kernel_sub << <gridSize, blockSize >> > (
		d_f1ptr, d_f2ptr, d_dt, H, W
	);

	cudaDeviceSynchronize();
}

__global__ void kernel_optflow(float* d_dx, float* d_dy, float* d_dt, float4* uv, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = rcToidx(row, col, H, W);
	if (row >= H - 5 || row <= 5 || col >= W-5 || col <= 5) {
		return;
	}

	float u_avg = 0.0f, v_avg = 0.0f;
	for (int iter = 0; iter < 8; iter++) {
		for (int i = -2; i <= 2; i++) {
			for (int j = -2; j <= 2; j++) {
				size_t indx = rcToidx(row + i, col + j, H, W);
				u_avg += uv[indx].x;
				v_avg += uv[indx].y;
			}
		}
		for (int i = -2; i <= 2; i++) {
			for (int j = -2; j <= 2; j++) {
				size_t indx = rcToidx(row + i, col + j, H, W);
				uv[indx].x = u_avg - d_dx[indx] * (d_dx[indx] * u_avg + d_dy[indx] * v_avg + d_dt[indx]) / (0.000001 + d_dx[indx] * d_dx[indx] + d_dy[indx] * d_dy[indx]);
				uv[indx].y = v_avg - d_dy[indx] * (d_dx[indx] * u_avg + d_dy[indx] * v_avg + d_dt[indx]) / (0.000001 + d_dx[indx] * d_dx[indx] + d_dy[indx] * d_dy[indx]);
			}
		}
	}
	__syncthreads();
}

void launch_optflow(float* d_dx, float* d_dy, float* d_dt, float4* uv, int H, int W) {

	const dim3 blockSize(32, 32);
	const dim3 gridSize(H / blockSize.x + 1, W / blockSize.y + 1);

	kernel_optflow << <gridSize, blockSize >> > (
		d_dx, d_dy, d_dt, uv, H, W
	);

	cudaError_t error = cudaDeviceSynchronize();
};