#define __CUDACC__
#include "CuCommon.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
__global__ void ScaleNearestNeighborForwardKernel(const __half* input, __half* output, const int batch, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth, const float scaleY, const float scaleX){
	const size_t total = static_cast<size_t>(batch)*channels*outHeight*outWidth;
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int outX = idx % outWidth;
		const int outY = idx/outWidth % outHeight;
		const int channel = idx/(static_cast<size_t>(outWidth)*outHeight) % channels;
		const int batchIndex = idx/(static_cast<size_t>(outWidth)*outHeight*channels);
		const int inY = min(static_cast<int>((outY + 0.5f)*scaleY), inHeight - 1);
		const int inX = min(static_cast<int>((outX + 0.5f)*scaleX), inWidth - 1);
		const size_t inIdx = ((static_cast<size_t>(batchIndex)*channels + channel)*inHeight + inY)*inWidth + inX;
		output[idx] = input[inIdx];
	}
}
void ScaleNearestNeighborForward(const __half* input, __half* output, const int batch, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth){
	const size_t total = static_cast<size_t>(batch)*channels*outHeight*outWidth;
	constexpr int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	const float scaleY = static_cast<float>(inHeight)/static_cast<float>(outHeight);
	const float scaleX = static_cast<float>(inWidth)/static_cast<float>(outWidth);
	ScaleNearestNeighborForwardKernel<<<blocks, bs>>>(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, scaleY, scaleX);
	checkCUDA(cudaGetLastError());
}
__global__ void ScaleNearestNeighborBackwardKernel(const __half* gradOut, __half* gradIn, const int batch, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth, const float scaleY, const float scaleX){
	const size_t total = static_cast<size_t>(batch)*channels*outHeight*outWidth;
	const auto stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int outX = idx % outWidth;
		const int outY = idx/outWidth % outHeight;
		const int channel = idx/(static_cast<size_t>(outWidth)*outHeight) % channels;
		const int batchIndex = idx/(static_cast<size_t>(outWidth)*outHeight*channels);
		const int inY = min(static_cast<int>((outY + 0.5f)*scaleY), inHeight - 1);
		const int inX = min(static_cast<int>((outX + 0.5f)*scaleX), inWidth - 1);
		const size_t inIdx = ((static_cast<size_t>(batchIndex)*channels + channel)*inHeight + inY)*inWidth + inX;
		atomicAdd(&gradIn[inIdx], gradOut[idx]);
	}
}
void ScaleNearestNeighborBackward(const __half* gradOut, __half* gradIn, const int batch, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth){
	const size_t total = static_cast<size_t>(batch)*channels*outHeight*outWidth;
	constexpr int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	const float scaleY = static_cast<float>(inHeight)/static_cast<float>(outHeight);
	const float scaleX = static_cast<float>(inWidth)/static_cast<float>(outWidth);
	ScaleNearestNeighborBackwardKernel<<<blocks, bs>>>(gradOut, gradIn, batch, channels, inHeight, inWidth, outHeight, outWidth, scaleY, scaleX);
	checkCUDA(cudaGetLastError());
}