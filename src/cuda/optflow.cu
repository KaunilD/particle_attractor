#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda/utils.hpp"

#define KERN_RADIUS 5

__device__ size_t GIDX(size_t row, size_t col, int H, int W) {
	return row * W + col;
}


__global__ void kernel_grey(
	float4* d_Iin,
	float* d_Iout,
	int numel
) {

	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	if (col >= numel) {
		return;
	}

	float4 pixel = d_Iin[col];

	d_Iout[col] = 0.2989f * (pixel.x) + 0.5870f * (pixel.y) + 0.1140f * (pixel.z);
}

void launch_gray(
	float4* d_Iin,
	float* d_Iout,
	int numel
) {


	dim3 blockSize = { 320 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(numel / blockSize.x))
	};

	kernel_grey << <gridSize, blockSize >> > (
		d_Iin, d_Iout, numel
		);

	CUDACHECK(cudaDeviceSynchronize());
}


__global__ void kernel_partials(
	float* d_f1ptr,
	float* d_f1dx, float* d_f1dy,
	int H, int W
) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;

	size_t idx = GIDX(row, col, H, W);
	if (row >= H || row <= 1 || col >= W || col <= 1) {
		return;
	}

	float gray_x1 = d_f1ptr[GIDX(row, col - 1, H, W)];
	float gray_x2 = d_f1ptr[GIDX(row, col + 1, H, W)];

	float gray_y1 = d_f1ptr[GIDX(row - 1, col, H, W)];
	float gray_y2 = d_f1ptr[GIDX(row + 1, col, H, W)];

	d_f1dx[idx] = (gray_x2 - gray_x1) / 2.0f;
	d_f1dy[idx] = (gray_y2 - gray_y1) / 2.0f;
}


void launch_partials(
	float* d_f1ptr,
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


__global__ void kernel_sub(float* d_f1ptr, float* d_f2ptr, float* d_dt, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = GIDX(row, col, H, W);

	if (row >= H || row <= 1 || col >= W || col <= 5) {
		return;
	}

	d_dt[idx] = d_f2ptr[idx] - d_f1ptr[idx];

}


void launch_sub(float* d_f1ptr, float* d_f2ptr, float* d_dt, int H, int W) {

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

__global__ void kernel_optflow(float* d_dx1, float* d_dy1, float* d_dx2, float* d_dy2, float* d_dt, float4* uv, float4* uv1, int H, int W) {

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

	uv1[idx].x = uv[idx].x;
	uv1[idx].y = uv[idx].y;

}

void launch_optflow(float* d_dx1, float* d_dy1, float* d_dx2, float* d_dy2, float* d_dt, float4* uv, float4* uv1, int H, int W) {

	dim3 blockSize = { 32, 32 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(W / blockSize.x)),
		static_cast<unsigned int>(ceil(H / blockSize.y))
	};

	kernel_optflow << <gridSize, blockSize >> > (
		d_dx1, d_dy1, d_dx2, d_dy2, d_dt, uv, uv1, H, W
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


__global__ void kernel_blur(float4* d_I, float4* d_Ib, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = GIDX(row, col, H, W);

	if (row >= H - KERN_RADIUS || row <= KERN_RADIUS || col >= W - KERN_RADIUS || col <= KERN_RADIUS) {
		return;
	}

	float a = 1 / 256.0;
	float b = 4 / 256.0;
	float c = 6 / 256.0;
	float d = 16 / 256.0;
	float e = 24 / 256.0;
	float f = 36 / 256.0;

	__shared__ float k[KERN_RADIUS * 2 + 1][KERN_RADIUS * 2 + 1];
	k[0][0] = a; k[1][0] = b; k[2][0] = c; k[3][0] = b; k[4][0] = a;
	k[0][1] = b; k[1][1] = d; k[2][1] = e; k[3][1] = d; k[4][1] = b;
	k[0][2] = c; k[1][2] = e; k[2][2] = f; k[3][2] = e; k[4][2] = c;
	k[0][3] = b; k[1][3] = d; k[2][3] = e; k[3][3] = d; k[4][3] = b;
	k[0][4] = a; k[1][4] = b; k[2][4] = c; k[3][4] = b; k[4][4] = a;


	d_Ib[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	for (int i = -KERN_RADIUS; i <= KERN_RADIUS; i++) {
		for (int j = -KERN_RADIUS; j <= KERN_RADIUS; j++) {
			d_Ib[idx].x += d_I[GIDX(row + i, col + j, H, W)].x * k[i + KERN_RADIUS][j + KERN_RADIUS];
			d_Ib[idx].y += d_I[GIDX(row + i, col + j, H, W)].y * k[i + KERN_RADIUS][j + KERN_RADIUS];
			d_Ib[idx].z += d_I[GIDX(row + i, col + j, H, W)].z * k[i + KERN_RADIUS][j + KERN_RADIUS];
		}
	}

}




void launch_blur(float4* d_I, float4* d_Ib, int H, int W) {

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


__global__ void kernel_convection(float4* d_uv1, float4* d_uv2, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = GIDX(row, col, H, W);

	if (row >= H - KERN_RADIUS || row <= KERN_RADIUS || col >= W - KERN_RADIUS || col <= KERN_RADIUS) {
		return;
	}

	d_uv2[idx].x = d_uv1[idx].x - (
		(0.2 * (d_uv1[idx].x - d_uv1[GIDX(row, col - 1, H, W)].x)) -
		(0.2 * (d_uv1[idx].x - d_uv1[GIDX(row - 1, col, H, W)].x))
	);

	d_uv2[idx].y = d_uv1[idx].y - (
		(0.2 * (d_uv1[idx].y - d_uv1[GIDX(row, col - 1, H, W)].y)) -
		(0.2 * (d_uv1[idx].y - d_uv1[GIDX(row - 1, col, H, W)].y))
	);


	d_uv1[idx].x = d_uv2[idx].x;
	d_uv1[idx].y = d_uv2[idx].y;
}





void launch_convection(float4* d_uv1, float4* d_uv2, int H, int W) {

	dim3 blockSize = { 32, 32 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(W / blockSize.x)),
		static_cast<unsigned int>(ceil(H / blockSize.y))
	};

	kernel_convection << <gridSize, blockSize >> > (
		d_uv1, d_uv2, H, W
		);

	CUDACHECK(cudaDeviceSynchronize());

};

__global__ void kernel_convert(uchar3* d_Iin, float4* d_Iout, int numel) {
	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if (col >= numel) { return; }
	uchar3 val = d_Iin[col];

	d_Iout[col] = make_float4(
		val.x / 255.0f,
		val.y / 255.0f,
		val.z / 255.0f,
		1.0f
	);
}

void launch_convert(uchar3* d_Iin, float4* d_Iout, int numel) {
	dim3 blockSize = { 320 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(numel / blockSize.x))
	};
	kernel_convert << <gridSize, blockSize >> > (
		d_Iin, d_Iout, numel
		);

	CUDACHECK(cudaDeviceSynchronize());
};
