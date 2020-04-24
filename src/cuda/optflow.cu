#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda/utils.hpp"

#define KERN_RADIUS 5

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

	float grad_x = (d_dx1[idx] + d_dx2[idx]) / 2.0f;
	float grad_y = (d_dy1[idx] + d_dy2[idx]) / 2.0f;

	__shared__ float k[9];
	k[0] = 0.084;
	k[1] = 0.167;
	k[2] = 0.084;
	k[3] = 0.167;
	k[4] = -1;
	k[5] = 0.167;
	k[6] = 0.084;
	k[7] = 0.167;
	k[8] = 0.084;


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
		static_cast<unsigned int>(ceil(numel / blockSize.x))
	};
	kernel_fill << <gridSize, blockSize >> > (
		d_dx1, val, numel
		);

	CUDACHECK(cudaDeviceSynchronize());
};

__global__ void kernel_blur(uchar3* d_I, uchar3* d_Ib, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = GIDX(row, col, H, W);

	if (row >= H - KERN_RADIUS || row <= KERN_RADIUS || col >= W - KERN_RADIUS || col <= KERN_RADIUS) {
		return;
	}

	__shared__ float k[KERN_RADIUS * 2 + 1][KERN_RADIUS * 2 + 1];
	k[0][0] = 1; k[1][0] = 4;  k[2][0] = 6;  k[3][0] = 4;  k[4][0] = 1;
	k[0][1] = 4; k[1][1] = 16; k[2][1] = 24; k[3][1] = 16; k[4][1] = 4;
	k[0][2] = 6; k[1][2] = 24; k[2][2] = 36; k[3][2] = 24; k[4][2] = 6;
	k[0][3] = 4; k[1][3] = 16; k[2][3] = 24; k[3][3] = 16; k[4][3] = 4;
	k[0][4] = 1; k[1][4] = 4;  k[2][4] = 6;  k[3][4] = 4;  k[4][4] = 1;

	for (int i = -KERN_RADIUS; i <= KERN_RADIUS; i++) {
		for (int j = -KERN_RADIUS; j <= KERN_RADIUS; j++) {
			d_Ib[GIDX(row, col, H, W)].x += d_I[GIDX(row + i, col + j, H, W)].x * k[i + KERN_RADIUS][j + KERN_RADIUS];
			d_Ib[GIDX(row, col, H, W)].y += d_I[GIDX(row + i, col + j, H, W)].y * k[i + KERN_RADIUS][j + KERN_RADIUS];
			d_Ib[GIDX(row, col, H, W)].z += d_I[GIDX(row + i, col + j, H, W)].z * k[i + KERN_RADIUS][j + KERN_RADIUS];
		}
	}

}


void launch_blur(uchar3* d_I, uchar3* d_Ib, int H, int W) {

	dim3 blockSize = { 32, 32 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(W / blockSize.x)),
		static_cast<unsigned int>(ceil(H / blockSize.y))
	};

	kernel_blur << <gridSize, blockSize >> > (
		d_I, d_Ib, H, W
		);

	CUDACHECK(cudaDeviceSynchronize());
};
