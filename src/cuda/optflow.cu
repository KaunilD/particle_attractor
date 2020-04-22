#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda/utils.hpp"

__device__ size_t GIDX(size_t row, size_t col, int H, int W) {
	return row * W + col;
}

__global__ void kernel_partials(
	uchar3* d_f1ptr,
	float* d_f1dx, float* d_f1dy,
	int H, int W
) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = GIDX(row, col, H, W);
	if (row >= H || row <= 1 || col >= W || col <= 1) {
		return;
	}

	//printf("%d %d %d\n", row, col, idx);
	float gray_x1 = 0.2989f * (d_f1ptr[GIDX(row, col - 1, H, W)].x / 255.0f) +
		0.5870f * (d_f1ptr[GIDX(row, col - 1, H, W)].y / 255.0f) +
		0.1140f * (d_f1ptr[GIDX(row, col - 1, H, W)].z / 255.0f);
	//printf("%d %d %f \n", row, col, gray_x1);
	float gray_x2 = 0.2989f * (d_f1ptr[GIDX(row, col + 1, H, W)].x / 255.0f) +
		0.5870f * (d_f1ptr[GIDX(row, col + 1, H, W)].y / 255.0f) +
		0.1140f * (d_f1ptr[GIDX(row, col + 1, H, W)].z / 255.0f);

	float gray_y1 = 0.2989f * (d_f1ptr[GIDX(row - 1, col, H, W)].x / 255.0f) +
		0.5870f * (d_f1ptr[GIDX(row - 1, col, H, W)].y / 255.0f) +
		0.1140f * (d_f1ptr[GIDX(row - 1, col, H, W)].z / 255.0f);

	float gray_y2 = 0.2989f * (d_f1ptr[GIDX(row + 1, col, H, W)].x / 255.0f) +
		0.5870f * (d_f1ptr[GIDX(row + 1, col, H, W)].y / 255.0f) +
		0.1140f * (d_f1ptr[GIDX(row + 1, col, H, W)].z / 255.0f);



	d_f1dx[idx] = (gray_x1 - gray_x2) / 2.0f;
    d_f1dy[idx] = (gray_y1 - gray_y2) / 2.0f;


}


void launch_partials(
	uchar3* d_f1ptr,
	float* d_f1dx, float* d_f1dy,
	int H, int W
) {


	dim3 blockSize = { 32, 32 };
	dim3 gridSize = {
	static_cast<unsigned int>(ceil(W / blockSize.x)),
	static_cast<unsigned int>(ceil(H / blockSize.y))
	};

	kernel_partials << <gridSize, blockSize >> > (
		d_f1ptr, d_f1dx, d_f1dy, H, W
		);

	CUDACHECK(cudaDeviceSynchronize());
}


__global__ void kernel_sub(uchar3* d_f1ptr, uchar3* d_f2ptr, float* d_dt, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = GIDX(row, col, H, W);

	if (row >= H || row <= 1 || col >= W || col <= 5) {
		return;
	}

	float gray_1 = 0.2989f * (d_f1ptr[idx].x / 255.0f) +
		0.5870f * (d_f1ptr[idx].y / 255.0f) +
		0.1140f * (d_f1ptr[idx].z / 255.0f);

	float gray_2 = 0.2989f * (d_f2ptr[idx].x / 255.0f) +
		0.5870f * (d_f2ptr[idx].y / 255.0f) +
		0.1140f * (d_f2ptr[idx].z / 255.0f);


	d_dt[idx] = gray_2 - gray_1;

}


void launch_sub(uchar3* d_f1ptr, uchar3* d_f2ptr, float* d_dt, int H, int W) {

	dim3 blockSize = { 32, 32 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(W / blockSize.x)),
		static_cast<unsigned int>(ceil(H / blockSize.y))
	};

	kernel_sub << <gridSize, blockSize >> > (
		d_f1ptr, d_f2ptr, d_dt, H, W
	);

	CUDACHECK(cudaDeviceSynchronize());
}

__global__ void kernel_optflow(float* d_dx1, float* d_dy1, float* d_dx2, float* d_dy2, float* d_dt, float4* uv, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = GIDX(row, col, H, W);
	
	if (row >= H || row <= 1 || col >= W || col <= 1) {
		return;
	}

	float grad_x = (d_dx1[idx] + d_dx2[idx])/2.0f;
	float grad_y = (d_dy1[idx] + d_dy2[idx])/2.0f;

	float k[9] = { 0.084, 0.167, 0.084, 0.167, -1, 0.167, 0.084, 0.167, 0.084 };

	float u_avg = uv[idx].x, v_avg = uv[idx].y, num = 0, denom = 0.01 + grad_x * grad_x + grad_y * grad_y;
	int count = 0;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			u_avg += uv[GIDX(row + i, col + j, H, W)].x * k[count];
			v_avg += uv[GIDX(row + i, col + j, H, W)].y * k[count];
			count++;

		}
	}
	num = (grad_x * u_avg + grad_y * v_avg + d_dt[idx]);
	
	uv[idx].x = u_avg - (grad_x * num) / denom;
	uv[idx].y = v_avg - (grad_y * num) / denom;
}

void launch_optflow(float* d_dx1, float* d_dy1, float* d_dx2, float* d_dy2, float* d_dt, float4* uv, int H, int W) {

	dim3 blockSize = { 32, 32 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(W / blockSize.x)),
		static_cast<unsigned int>(ceil(H / blockSize.y))
	};

	kernel_optflow << <gridSize, blockSize >> > (
		d_dx1, d_dy1, d_dx2, d_dy2, d_dt, uv, H, W
		);

	CUDACHECK(cudaDeviceSynchronize());
};

__global__ void kernel_fill(float4* d_dx1, float val, int numel) {
	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if (col >= numel) { return; }
	
	d_dx1[col] = make_float4(val, val, val, val);
}

void launch_fill(float4* d_dx1, float val, int numel) {
	dim3 blockSize = { 320 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(numel  / blockSize.x))
	};
	kernel_fill << <gridSize, blockSize >> > (
		d_dx1, val, numel
		);

	CUDACHECK(cudaDeviceSynchronize());
};