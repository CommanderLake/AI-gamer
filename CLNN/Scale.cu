#define __CUDACC__
#include "CuCommon.h"
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
__global__ void ScaleNearestNeighborLetterboxForwardKernel(const __half* input, __half* output, const int batch, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth, const int contentTop, const int contentLeft, const int contentHeight, const int contentWidth, const float scaleY, const float scaleX){
	const size_t total = static_cast<size_t>(batch)*channels*outHeight*outWidth;
	const auto stride = blockDim.x*gridDim.x;
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int outX = idx % outWidth;
		const int outY = idx/outWidth % outHeight;
		if(outY < contentTop || outY >= contentTop + contentHeight || outX < contentLeft || outX >= contentLeft + contentWidth){
			output[idx] = __float2half(0.0f);
			continue;
		}
		const int channel = idx/(static_cast<size_t>(outWidth)*outHeight) % channels;
		const int batchIndex = idx/(static_cast<size_t>(outWidth)*outHeight*channels);
		const int contentY = outY - contentTop;
		const int contentX = outX - contentLeft;
		const int inY = min(static_cast<int>((contentY + 0.5f)*scaleY), inHeight - 1);
		const int inX = min(static_cast<int>((contentX + 0.5f)*scaleX), inWidth - 1);
		const size_t inIdx = ((static_cast<size_t>(batchIndex)*channels + channel)*inHeight + inY)*inWidth + inX;
		output[idx] = input[inIdx];
	}
}
void ScaleNearestNeighborLetterboxForward(const __half* input, __half* output, const int batch, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth, const int contentTop, const int contentLeft, const int contentHeight, const int contentWidth){
	const size_t total = static_cast<size_t>(batch)*channels*outHeight*outWidth;
	constexpr int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	const float scaleY = static_cast<float>(inHeight)/static_cast<float>(contentHeight);
	const float scaleX = static_cast<float>(inWidth)/static_cast<float>(contentWidth);
	ScaleNearestNeighborLetterboxForwardKernel<<<blocks, bs>>>(input, output, batch, channels, inHeight, inWidth, outHeight, outWidth, contentTop, contentLeft, contentHeight, contentWidth, scaleY, scaleX);
	checkCUDA(cudaGetLastError());
}
__global__ void ScaleNearestNeighborLetterboxBackwardKernel(const __half* gradOut, __half* gradIn, const int batch, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth, const int contentTop, const int contentLeft, const int contentHeight, const int contentWidth, const float scaleY, const float scaleX){
	const size_t total = static_cast<size_t>(batch)*channels*contentHeight*contentWidth;
	const auto stride = blockDim.x*gridDim.x;
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int contentX = idx % contentWidth;
		const int contentY = idx/contentWidth % contentHeight;
		const int channel = idx/(static_cast<size_t>(contentWidth)*contentHeight) % channels;
		const int batchIndex = idx/(static_cast<size_t>(contentWidth)*contentHeight*channels);
		const int inY = min(static_cast<int>((contentY + 0.5f)*scaleY), inHeight - 1);
		const int inX = min(static_cast<int>((contentX + 0.5f)*scaleX), inWidth - 1);
		const size_t inIdx = ((static_cast<size_t>(batchIndex)*channels + channel)*inHeight + inY)*inWidth + inX;
		const int outY = contentTop + contentY;
		const int outX = contentLeft + contentX;
		const size_t outIdx = ((static_cast<size_t>(batchIndex)*channels + channel)*outHeight + outY)*outWidth + outX;
		atomicAdd(&gradIn[inIdx], gradOut[outIdx]);
	}
}
void ScaleNearestNeighborLetterboxBackward(const __half* gradOut, __half* gradIn, const int batch, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth, const int contentTop, const int contentLeft, const int contentHeight, const int contentWidth){
	const size_t total = static_cast<size_t>(batch)*channels*contentHeight*contentWidth;
	constexpr int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	const float scaleY = static_cast<float>(inHeight)/static_cast<float>(contentHeight);
	const float scaleX = static_cast<float>(inWidth)/static_cast<float>(contentWidth);
	ScaleNearestNeighborLetterboxBackwardKernel<<<blocks, bs>>>(gradOut, gradIn, batch, channels, inHeight, inWidth, outHeight, outWidth, contentTop, contentLeft, contentHeight, contentWidth, scaleY, scaleX);
	checkCUDA(cudaGetLastError());
}
