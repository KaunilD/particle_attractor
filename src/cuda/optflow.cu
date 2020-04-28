#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "cuda/utils.hpp"

#define KERN_RADIUS 5

__constant__ float gaussian_kernel[KERN_RADIUS * KERN_RADIUS];
__constant__ float hs_kernel[9];

__device__ size_t GIDX(size_t row, size_t col, int H, int W) {
	return row * W + col;
}

void setGaussian(float const* elements, const int count) {
	if (count != 25) {
		std::cerr << "Only 5x5 gaussian kernel supported!" << std::endl;
		return;
	}
	CUDACHECK(cudaMemcpyToSymbol(gaussian_kernel, elements, count * sizeof(float)));
}

void setHS(float const* elements, const int count) {
	if (count != 9) {
		std::cerr << "Only 3x3 horn-schunke kernel supported!" << std::endl;
		return;
	}
	CUDACHECK(cudaMemcpyToSymbol(hs_kernel, elements, count * sizeof(float)));
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

	if (row >= H || col >= W) {
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

	const size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	const size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	const size_t idx = GIDX(row, col, H, W);

	const size_t s_tidx = threadIdx.y + 1; // 1 padding for HS Kernel
	const size_t s_tidy = threadIdx.y + 1; // 1 padding for HS Kernel

	extern __shared__ float4 s_img[];


	if (row >= H || row <= 1 || col >= W || col <= 1) {
		return;
	}

	s_img[GIDX(s_tidy, s_tidx, 34, 34)] = uv[idx];

	if (threadIdx.x == 0) {
		s_img[GIDX(s_tidy, s_tidx - 1, 34, 34)] = uv[GIDX(row, col - 1, H, W)];
	}

	if (threadIdx.x == blockDim.x - 1) {
		s_img[GIDX(s_tidy, s_tidx + 1, 34, 34)] = uv[GIDX(row, col + 1, H, W)];
	}

	if (threadIdx.y == 0) {
		s_img[GIDX(s_tidy - 1, s_tidx, 34, 34)] = uv[GIDX(row - 1, col - 1, H, W)];
	}

	if (threadIdx.x == blockDim.y - 1) {
		s_img[GIDX(s_tidy+1, s_tidx, 34, 34)] = uv[GIDX(row + 1, col + 1, H, W)];
	}

	__syncthreads();

	float grad_x = (d_dx1[idx] + d_dx2[idx]) / 2.0f;
	float grad_y = (d_dy1[idx] + d_dy2[idx]) / 2.0f;


	float u_avg = s_img[GIDX(s_tidy, s_tidx, 34, 34)].x, v_avg = s_img[GIDX(s_tidy, s_tidx, 34, 34)].y, num = 0, denom = 0.01 + grad_x * grad_x + grad_y * grad_y;
	int count = 0;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			u_avg += s_img[GIDX(s_tidy + i, s_tidx + j, 34, 34)].x * hs_kernel[count];
			v_avg += s_img[GIDX(s_tidy + i, s_tidx + j, 34, 34)].y * hs_kernel[count];
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
	size_t sharedMem = sizeof(float4) * (blockSize.y + 2) * (blockSize.x + 2);

	const float k[9] = { 0.084, 0.167, 0.084, 0.167, -1, 0.167, 0.084, 0.167, 0.084 };

	setHS(k, 9);

	kernel_optflow << <gridSize, blockSize, sharedMem >> > (
		d_dx1, d_dy1, d_dx2, d_dy2, d_dt, uv, uv1, H, W
		);

	CUDACHECK(cudaDeviceSynchronize());
};

__global__ void kernel_fill(float4* d_dx1, float val, int numel) {
	size_t col = threadIdx.x + blockIdx.x * blockDim.x;
	if (col >= numel) { return; }

	d_dx1[col].x /= val;
	d_dx1[col].y /= val;
	d_dx1[col].z /= val;
	d_dx1[col].w /= val;
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


__global__ void kernel_blur(float* d_I, float* d_Ib, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = GIDX(row, col, H, W);

	if (row >= H - KERN_RADIUS || row <= KERN_RADIUS || col >= W - KERN_RADIUS || col <= KERN_RADIUS) {
		return;
	}


	int count = 0;
	for (int i = -KERN_RADIUS; i <= KERN_RADIUS; i++) {
		for (int j = -KERN_RADIUS; j <= KERN_RADIUS; j++) {
			d_Ib[idx] += d_I[GIDX(row + i, col + j, H, W)] * gaussian_kernel[count];
			count++;
		}
	}

}




void launch_blur(float* d_I, float* d_Ib, int H, int W) {

	dim3 blockSize = { 32, 32 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(W / blockSize.x)),
		static_cast<unsigned int>(ceil(H / blockSize.y))
	};

	float a = 1 / 256.0;
	float b = 4 / 256.0;
	float c = 6 / 256.0;
	float d = 16 / 256.0;
	float e = 24 / 256.0;
	float f = 36 / 256.0;

	const float k[25] = {
		a, b, c, b, a,
		b, d, e, d, b,
		c, e, f, e, c,
		b, d, e, d, b,
		a, b, c, b, a
	};

	setGaussian(k, 25);

	kernel_blur << <gridSize, blockSize >> > (
		d_I, d_Ib, H, W
		);

	CUDACHECK(cudaDeviceSynchronize());
};


__global__ void kernel_convection(float4* d_uv1, float4* d_uv2, float* d_p, int H, int W) {

	size_t row = threadIdx.y + blockDim.y * blockIdx.y;
	size_t col = threadIdx.x + blockDim.x * blockIdx.x;
	size_t idx = GIDX(row, col, H, W);


	if (row >= H - 1 || row <= 1 || col >= W - 1 || col <= 1) {
		return;
	}

	float dt = 0.0001;
	float nu = 0.1;
	float dx = 1, dy = 1, dx2 = dx*dx, dy2 = dy*dy;
	float rho = 1.2;
	
	float4 d_uv1r1 = d_uv1[GIDX(row - 1, col, H, W)], d_uv1r2 = d_uv1[GIDX(row + 1, col, H, W)];
	float4 d_uv1c1 = d_uv1[GIDX(row, col - 1, H, W)], d_uv1c2 = d_uv1[GIDX(row, col + 1, H, W)];
	float4 d_uv1_idx = d_uv1[idx];

	float d_pr1 = d_p[GIDX(row - 1, col, H, W)], d_pr2 = d_p[GIDX(row + 1, col, H, W)];
	float d_pc1 = d_p[GIDX(row, col - 1, H, W)], d_pc2 = d_p[GIDX(row, col + 1, H, W)];
	float d_p_idx = d_p[idx];
	float a = (d_pr2 + d_pr1) * dy2;
	float b = (d_pc2 + d_pc1) * dx2;
	float c = 2 * (dx2 + dy2);
	float d = rho * dx2 * dy2;

	d_p[idx] = ((a + b) / c) - (d / c) *
		(1 / dt * ((d_uv1r2.x - d_uv1r1.x) / (2 * dx) + (d_uv1c2.y - d_uv1c1.y) / (2 * dy)) -
			((d_uv1r2.x - d_uv1r1.x) / (2 * dx)) * ((d_uv1c2.y - d_uv1c1.y) / (2 * dy)) -
			(2 * (d_uv1c2.x - d_uv1c1.x) / (2 * dy)) * (2 * (d_uv1r2.y - d_uv1r1.y) / (2 * dx)) -
			(2 * (d_uv1c2.y - d_uv1c1.y) / (2 * dy)) * (2 * (d_uv1c2.y - d_uv1c1.y) / (2 * dx)));


	d_uv2[idx].x = d_uv1_idx.x -
		d_uv1_idx.x * dt * (d_uv1_idx.x - d_uv1r1.x) -
		d_uv1_idx.y * dt * (d_uv1_idx.x - d_uv1c1.x) -
		(dt / rho) * (d_pr2 - d_pr1) + 
		nu * (dt * (d_uv1r2.x - 2 * d_uv1[idx].x + d_uv1r1.x) + dt * (d_uv1c2.x - 2 * d_uv1_idx.x + d_uv1c1.x));

	d_uv2[idx].y = d_uv1_idx.y -
		d_uv1_idx.x * dt * (d_uv1_idx.y - d_uv1r1.y) -
		d_uv1_idx.y * dt * (d_uv1_idx.y - d_uv1c1.y) -
		(dt / rho) * (d_pr2 - d_pr1) +
		nu * (dt * (d_uv1r2.y - 2 * d_uv1_idx.y + d_uv1r1.y) +
			dt * (d_uv1c2.y - 2 * d_uv1_idx.y + d_uv1c1.y));


	
	d_uv1[idx].x = d_uv2[idx].x;
	d_uv1[idx].y = d_uv2[idx].y;

}





void launch_convection(float4* d_uv1, float4* d_uv2, float*p, int H, int W) {

	dim3 blockSize = { 32, 32 };
	dim3 gridSize = {
		static_cast<unsigned int>(ceil(W / blockSize.x)),
		static_cast<unsigned int>(ceil(H / blockSize.y))
	};
	int padding = 1;

	size_t s_numel = (blockSize.x + 2 * padding) * (blockSize.y + 2 * padding);
	size_t s_uv1 = s_numel * sizeof(float4);
	size_t s_p = s_numel * sizeof(float);

	kernel_convection << <gridSize, blockSize, s_uv1 + s_p >> > (
		d_uv1, d_uv2, p, H, W
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
