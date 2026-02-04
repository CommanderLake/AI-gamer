#define __CUDACC__
#include "CuCommon.cuh"
#include <cuda.h>
#include <curand.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
void BlockShiftHalf(__half* hPtr, const int shiftBy, const int blocksToShift){
	if(!hPtr || shiftBy == 0 || blocksToShift <= 1){
		return;
	}
	auto blockSize = shiftBy;
	if(blockSize < 0) blockSize = -blockSize;
	if(shiftBy > 0){
		for(int i = blocksToShift - 1; i > 0; --i){
			checkCUDA(cudaMemcpy(hPtr + i*blockSize, hPtr + (i - 1)*blockSize, blockSize*sizeof(__half), cudaMemcpyDeviceToDevice));
		}
	} else{
		for(int i = 0; i + 1 < blocksToShift; ++i){
			checkCUDA(cudaMemcpy(hPtr + (i - 1)*blockSize, hPtr + i*blockSize, blockSize*sizeof(__half), cudaMemcpyDeviceToDevice));
		}
	}
}
__device__ int deviceResult;
__global__ void isNaNKernel(const __half* __restrict__ data, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size && __hisnan(data[idx])){ atomicExch(&deviceResult, 1); }
}
bool IsnanHalf(const __half* __restrict__ data, int size){
	int hResult = 0;
	cudaMemcpyToSymbol(deviceResult, &hResult, sizeof(int));
	auto gridSize = DivCeil(size, BS);
	isNaNKernel<<<gridSize, BS>>>(data, size);
	checkCUDA(cudaGetLastError());
	cudaMemcpyFromSymbol(&hResult, deviceResult, sizeof(int));
	return hResult != 0;
}
__global__ void FeatureMapMosaicKernel(const __half* __restrict__ input, unsigned char* __restrict__ output, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, const float scale){
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int x = blockIdx.z*blockDim.z + threadIdx.z;
	if(c >= inC || y >= H || x >= W) return;
	const int tileX = c % gridW;
	const int tileY = c/gridW;
	const int outX = tileX*tileW + x;
	const int outY = tileY*tileH + y;
	const float fVal = __half2float(input[c*H*W + y*W + x]);
	const unsigned char pixel = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, fVal*255.0f*scale)));
	output[outY*mosaicW + outX] = pixel;
}
void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, const float scale, cudaStream_t stream){
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((inC + blockDim.x - 1)/blockDim.x, (H + blockDim.y - 1)/blockDim.y, (W + blockDim.z - 1)/blockDim.z);
	FeatureMapMosaicKernel<<<gridDim, blockDim, 0, stream>>>(dInput, dOutput, H, W, inC, mosaicW, tileW, tileH, gridW, scale);
	checkCUDA(cudaGetLastError());
}
__global__ void ScaleHalfKernel(__half* data, const size_t count, const float scale){
	const size_t stride = static_cast<size_t>(blockDim.x)*gridDim.x;
	for(size_t idx = blockIdx.x*blockDim.x + threadIdx.x; idx < count; idx += stride){ data[idx] = __float2half(__half2float(data[idx])*scale); }
}
void ScaleArrayHalf(__half* data, const size_t count, const float scale){
	if(!data || scale == 1.0f || count == 0) return;
	constexpr int bs = 256;
	const auto blocks = DivCeil(count, bs);
	ScaleHalfKernel<<<blocks, bs>>>(data, count, scale);
	checkCUDA(cudaGetLastError());
}
__global__ void AddBiasKernel(__half* output, const __half* bias, const int channels, const int batchSize){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = channels*batchSize;
	if(idx >= total){ return; }
	const int c = idx % channels;
	output[idx] = output[idx] + bias[c];
}
void AddBias(__half* output, const __half* bias, const int channels, const int batchSize){
	constexpr int bs = 256;
	const auto blocks = DivCeil(channels*batchSize, bs);
	AddBiasKernel<<<blocks, bs>>>(output, bias, channels, batchSize);
	checkCUDA(cudaGetLastError());
}
__global__ void AddTensorKernel(__half alpha, __half* A, __half beta, const __half* B, const int size){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		A[idx] = __hfma(alpha, A[idx], __hfma(beta, B[idx], 0));
	}
}
void AddTensor(float alpha, __half* A, float beta, const __half* B, const int size){
	size_t blocks, threads = 256;
	GetLaunchConfigGridStride(size, blocks, threads);
	AddTensorKernel<<<blocks, threads>>>(__half(alpha), A, __half(beta), B, size);
	checkCUDA(cudaGetLastError());
}
__global__ void AccumulateBiasGradKernel(const __half* grad, __half* gradBias, const int channels, const int batch, const float scale, const bool reset){
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	if(c >= channels){ return; }
	float sum = 0.0f;
	for(int b = 0; b < batch; ++b){ sum += __half2float(grad[c + b*channels]); }
	const float scaled = sum*scale;
	if(reset){ gradBias[c] = __float2half(scaled); } else{ gradBias[c] = __float2half(__half2float(gradBias[c]) + scaled); }
}
void AccumulateBiasGrad(const __half* grad, __half* gradBias, const int channels, const int batch, const float scale, const bool reset){
	constexpr int bs = 256;
	const auto blocks = DivCeil(channels, bs);
	AccumulateBiasGradKernel<<<blocks, bs>>>(grad, gradBias, channels, batch, scale, reset);
	checkCUDA(cudaGetLastError());
}