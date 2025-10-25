#include "CuCommon.cuh"
#include <device_launch_parameters.h>
__global__ void cuARGBtoRGB(const pixARGB* src, pixRGB* dst, int n){
	const auto stride = blockDim.x*gridDim.x;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride){
		dst[i].R = src[i].R;
		dst[i].G = src[i].G;
		dst[i].B = src[i].B;
	}
}
void ARGBtoRGB(unsigned char* src, unsigned char* dst, int n){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(n, blocks, tpb);
	cuARGBtoRGB<<<blocks, tpb>>>(reinterpret_cast<pixARGB*>(src), reinterpret_cast<pixRGB*>(dst), n);
}
__global__ void cuARGBtoRGBplanar(const unsigned char* src, unsigned char* dst, int n){
	const auto stride = blockDim.x*gridDim.x;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride){
		const int srcIdx = i*4;
		dst[i] = src[srcIdx + 2];
		dst[i + n] = src[srcIdx + 1];
		dst[i + 2*n] = src[srcIdx];
	}
}
void ARGBtoRGBplanar(unsigned char* src, unsigned char* dst, int n){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(n, blocks, tpb);
	cuARGBtoRGBplanar<<<blocks, tpb>>>(src, dst, n);
}
__global__ void ConvertByteToHalfKernel(const unsigned char* input, __half* output, const size_t size, const float scale){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = __float2half(input[idx]/scale); }
}
void ConvertByteToHalf(const unsigned char* input, __half* output, const size_t size, bool normalize){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	ConvertByteToHalfKernel<<<blocks, tpb>>>(input, output, size, normalize ? 255.0 : 1.0f);
}
__global__ void ConvertHalfToByteKernel(const __half* input, unsigned char* output, const size_t size, const float scale){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = static_cast<unsigned char>(__half2float(input[idx])*scale); }
}
void ConvertHalfToByte(const __half* input, unsigned char* output, const size_t size, const bool normalize){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	ConvertHalfToByteKernel<<<blocks, tpb>>>(input, output, size, normalize ? 255.0f : 1.0f);
}
__global__ void ConvertFloatToHalfKernel(const float* input, __half* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = __float2half(input[idx]); }
}
void ConvertFloatToHalf(const float* input, __half* output, const size_t size){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	ConvertFloatToHalfKernel<<<blocks, tpb>>>(input, output, size);
}
__global__ void ConvertHalfToFloatKernel(const __half* input, float* output, const size_t size){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ output[idx] = __half2float(input[idx]); }
}
void ConvertHalfToFloat(const __half* input, float* output, const size_t size){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	ConvertHalfToFloatKernel<<<blocks, tpb>>>(input, output, size);
}
__global__ void ConvertFloatToHalfScaleKernel(__half* halfWeights, const float* weights, const size_t size, const float scale){
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){ halfWeights[idx] = __float2half(weights[idx]*scale); }
}
void ConvertFloatToHalfScale(__half* halfWeights, const float* weights, const size_t size, const float scale){
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(size, blocks, tpb);
	ConvertFloatToHalfScaleKernel<<<blocks, tpb>>>(halfWeights, weights, size, scale);
}