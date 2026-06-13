#define __CUDACC__
#include "CuCommon.h"
#include <device_launch_parameters.h>
#define LOOP(i,a,b) for(size_t i=(a)+size_t(blockIdx.x)*blockDim.x+threadIdx.x,s=size_t(blockDim.x)*gridDim.x;i<(b);i+=s)
__device__ __forceinline__ unsigned char u8(float x){ return static_cast<unsigned char>(__float2uint_rz(x)); }
__global__ void cuARGBtoRGB(const PixARGB* src, PixRGB* dst, size_t n){
	const auto stride = blockDim.x*gridDim.x;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride){
		dst[i].R = src[i].R;
		dst[i].G = src[i].G;
		dst[i].B = src[i].B;
	}
}
void ARGBtoRGB(unsigned char* src, unsigned char* dst, size_t n){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(n, blocks, tpb);
	cuARGBtoRGB<<<blocks, tpb>>>(reinterpret_cast<PixARGB*>(src), reinterpret_cast<PixRGB*>(dst), n);
	checkCUDA(cudaGetLastError());
}
__global__ void cuARGBtoRGBplanar(const unsigned char* src, unsigned char* dst, size_t n){
	const auto stride = blockDim.x*gridDim.x;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride){
		const int srcIdx = i*4;
		dst[i] = src[srcIdx + 2];
		dst[i + n] = src[srcIdx + 1];
		dst[i + 2*n] = src[srcIdx];
	}
}
void ARGBtoRGBplanar(const unsigned char* src, unsigned char* dst, size_t n){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(n, blocks, tpb);
	cuARGBtoRGBplanar<<<blocks, tpb>>>(src, dst, n);
	checkCUDA(cudaGetLastError());
}
__global__ void B2H(const unsigned char* __restrict__ x, __half* __restrict__ y, const size_t n, const float inv){
	const size_t n4 = n >> 2;
	const auto x4 = reinterpret_cast<const uchar4*>(x);
	const auto y2 = reinterpret_cast<__half2*>(y);
	LOOP(i, 0, n4){
		const uchar4 v = x4[i];
		y2[2*i] = __floats2half2_rn(v.x*inv, v.y*inv);
		y2[2*i + 1] = __floats2half2_rn(v.z*inv, v.w*inv);
	}
	LOOP(i, n4<<2, n) y[i] = __float2half(x[i]*inv);
}
void ConvertByteToHalf(const unsigned char* input, __half* output, const size_t size, const bool normalize){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	B2H<<<blocks, tpb>>>(input, output, size, normalize ? 1.0f/255.0f : 1.0f);
	checkCUDA(cudaGetLastError());
}
__global__ void H2B(const __half* __restrict__ x, unsigned char* __restrict__ y, const size_t n, const float scale){
	const size_t n4 = n >> 2;
	const auto x2 = reinterpret_cast<const __half2*>(x);
	const auto y4 = reinterpret_cast<uchar4*>(y);
	LOOP(i, 0, n4){
		const float2 a = __half22float2(x2[2*i]), b = __half22float2(x2[2*i + 1]);
		y4[i] = make_uchar4(u8(a.x*scale), u8(a.y*scale), u8(b.x*scale), u8(b.y*scale));
	}
	LOOP(i, n4<<2, n) y[i] = u8(__half2float(x[i])*scale);
}
void ConvertHalfToByte(const __half* input, unsigned char* output, const size_t size, const bool normalize){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	H2B<<<blocks, tpb>>>(input, output, size, normalize ? 255.0f : 1.0f);
	checkCUDA(cudaGetLastError());
}
__global__ void F2H(const float* __restrict__ x, __half* __restrict__ y, size_t n){
	const size_t n4 = n >> 2;
	const auto x4 = reinterpret_cast<const float4*>(x);
	const auto y2 = reinterpret_cast<__half2*>(y);
	LOOP(i, 0, n4){
		const float4 v = x4[i];
		y2[2*i] = __floats2half2_rn(v.x, v.y);
		y2[2*i + 1] = __floats2half2_rn(v.z, v.w);
	}
	LOOP(i, n4<<2, n) y[i] = __float2half(x[i]);
}
void ConvertFloatToHalf(const float* input, __half* output, const size_t size){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	F2H<<<blocks, tpb>>>(input, output, size);
	checkCUDA(cudaGetLastError());
}
__global__ void H2F(const __half* __restrict__ x, float* __restrict__ y, size_t n){
	const size_t n4 = n >> 2;
	const auto x2 = reinterpret_cast<const __half2*>(x);
	const auto y4 = reinterpret_cast<float4*>(y);
	LOOP(i, 0, n4){
		const float2 a = __half22float2(x2[2*i]), b = __half22float2(x2[2*i + 1]);
		y4[i] = make_float4(a.x, a.y, b.x, b.y);
	}
	LOOP(i, n4<<2, n) y[i] = __half2float(x[i]);
}
void ConvertHalfToFloat(const __half* input, float* output, const size_t size){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	H2F<<<blocks, tpb>>>(input, output, size);
	checkCUDA(cudaGetLastError());
}
__global__ void F2HS(__half* __restrict__ y, const float* __restrict__ x, const size_t n, const float scale){
	const size_t n4 = n >> 2;
	const auto x4 = reinterpret_cast<const float4*>(x);
	const auto y2 = reinterpret_cast<__half2*>(y);
	LOOP(i, 0, n4){
		const float4 v = x4[i];
		y2[2*i] = __floats2half2_rn(v.x*scale, v.y*scale);
		y2[2*i + 1] = __floats2half2_rn(v.z*scale, v.w*scale);
	}
	LOOP(i, n4<<2, n) y[i] = __float2half(x[i]*scale);
}
void ConvertFloatToHalfScale(__half* halfWeights, const float* weights, const size_t size, const float scale){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	F2HS<<<blocks, tpb>>>(halfWeights, weights, size, scale);
	checkCUDA(cudaGetLastError());
}