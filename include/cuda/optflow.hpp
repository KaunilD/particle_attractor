#ifndef OPTFLOW_H
#define OPTFLOW_H

#include "libs.hpp"
#include "cuda/utils.hpp"

/*
	@ d_ structure pointers in CUDA
	@ h_ structure pointers on CPU
*/

class OpticalFlow {
public:
	/*
		useful constants for bound checking within 
		GPU
	*/
	int m_H, m_W, m_numpixels;
	
	/*
		device pointers for images taken at different
		time intervals.
	*/
	uchar3* d_f1ptr, * d_f2ptr;
	/*
		
	*/
	float* d_f1dx, * d_f1dy;
	float* d_f2dx, * d_f2dy;
	float* d_dt;

	float4* d_uv;

	OpticalFlow() = default;
	OpticalFlow(int H, int W) : m_H(H), m_W(W), m_numpixels(H* W) {
		LOG("OpticalFlow::c_tor()");

		CUDACHECK(cudaMalloc(&d_f1ptr, m_numpixels * sizeof(uchar3)));
		CUDACHECK(cudaMalloc(&d_f2ptr, m_numpixels * sizeof(uchar3)));

		CUDACHECK(cudaMalloc(&d_f1dx, m_numpixels * sizeof(float)));
		CUDACHECK(cudaMalloc(&d_f1dy, m_numpixels * sizeof(float)));

		CUDACHECK(cudaMalloc(&d_f2dx, m_numpixels * sizeof(float)));
		CUDACHECK(cudaMalloc(&d_f2dy, m_numpixels * sizeof(float)));

		CUDACHECK(cudaMalloc(&d_uv, m_numpixels * sizeof(float4)));

		CUDACHECK(cudaMalloc(&d_dt, m_numpixels * sizeof(float)));
	}

	void copy(const cv::Mat& f1, const cv::Mat& f2) {
		CUDACHECK(cudaMemcpy(d_f1ptr, (uchar3*)f1.ptr<unsigned char>(), m_numpixels * sizeof(uchar3), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(d_f2ptr, (uchar3*)f2.ptr<unsigned char>(), m_numpixels * sizeof(uchar3), cudaMemcpyHostToDevice));
	}

	~OpticalFlow() {
		cudaFree(d_f1ptr);
		cudaFree(d_f2ptr);

		cudaFree(d_f1dx);
		cudaFree(d_f1dy);

		cudaFree(d_f2dx);
		cudaFree(d_f2dy);

		cudaFree(d_uv);

		cudaFree(d_dt);
	
	}
};

/*
	Prototype declarations for their respective .cu kernerl. 
	This makes the linker happy.
*/


/*
 Computes discrete partial derevatives of the 2D image function I 
 in x and y direction.
 The gradient is computed using first differences at the borders (i) 
 and using centeral differences at the interior (ii).
 i	= del f(x) = f(x + 1) - f(x)
 ii	= del f(x) = f(x + 1) - f(x - 1)
 @input		= uchar3* image intensities.
 @output	= float*
*/
void launch_partials(
	uchar3* d_I, 
	float* d_, float* , 
	int H, int W
);

void launch_sub(
	uchar3* d_f1ptr, uchar3* d_f2ptr, 
	float* d_dt, 
	int H, int W
);
void launch_fill(
	float4* d_uv, 
	float val, 
	int numel
);
void launch_add(
	float1* a, float1* b, 
	float1* c, 
	int H, int W
);
void launch_prod(
	float1* a, float1* b, 
	float1* c, 
	int H, int W
);
void launch_optflow(
	float* d_dx1, float* d_dy1, float* d_dx2, float* d_dy2, float* d_dt, 
	float4* uv, 
	int H, int W

);
#endif OPTFLOW_H