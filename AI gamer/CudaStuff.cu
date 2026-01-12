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
__global__ void GradientKernel(__half* grads, const __half* predictions, const __half* targets, const float clip, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){ grads[idx] = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx] - targets[idx])))); }
}
void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const float clip, const int size){
	auto gridSize = DivCeil(size, BS);
	GradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, size);
	checkCUDA(cudaGetLastError());
}
__device__ inline float Sigmoidf(const float x){
	if(x >= 0.0f){
		const float z = __expf(-x);
		return 1.0f / (1.0f + z);
	}
	const float z = __expf(x);
	return z / (1.0f + z);
}
__global__ void LossBackpropKernel(__half* gradients, const __half* predictions, const float* targets, const float clip, const int numCtrls, const int numButs, const int batchSize, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx / numCtrls;
		const int ctrlId = idx % numCtrls;
		const float target = targets[idx];
		if(ctrlId < numButs){
			const float logit = __half2float(predictions[idx]);
			const float prob = Sigmoidf(logit);
			gradients[batchId*numButs + ctrlId] = __float2half(fmaxf(-clip, fminf(clip, prob - target)));
		} else{
			const float pred = __half2float(predictions[idx]);
			gradients[numButs*batchSize + batchId*(numCtrls - numButs) + (ctrlId - numButs)] = __float2half(fmaxf(-clip, fminf(clip, pred - target)));
		}
	}
}
void LossBackprop(__half* dGradient, const __half* dPredictions, const float* dTargets, const float clip, const int size, const int numCtrls, const int numButs, const int batchSize){
	auto gridSize = DivCeil(size, BS);
	LossBackpropKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, numCtrls, numButs, batchSize, size);
	checkCUDA(cudaGetLastError());
}
__global__ void MergeOutputsKernel(__half* predOut, const __half* buttonData, const __half* axisData, const int size, const int numCtrls, const int numButs){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx / numCtrls;
		const int ctrlId = idx % numCtrls;
		if(ctrlId < numButs){ predOut[idx] = buttonData[batchId*numButs + ctrlId]; } else{ predOut[idx] = axisData[batchId*(numCtrls - numButs) + (ctrlId - numButs)]; }
	}
}
void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, const int numCtrls, const int numButs, const int size){
	auto gridSize = DivCeil(size, BS);
	MergeOutputsKernel<<<gridSize, BS>>>(predOut, buttonData, axisData, size, numCtrls, numButs);
	checkCUDA(cudaGetLastError());
}
__global__ void BCEGradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size, const float scale){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size) gradients[idx] = __float2half((Sigmoidf(__half2float(predictions[idx])) - __half2float(targets[idx]))*scale);
}
void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size, const float scale){
	auto gridSize = DivCeil(size, BS);
	BCEGradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, scale);
	checkCUDA(cudaGetLastError());
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
	const int tileY = c / gridW;
	const int outX = tileX*tileW + x;
	const int outY = tileY*tileH + y;
	const float fVal = __half2float(input[c*H*W + y*W + x]);
	const unsigned char pixel = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, fVal*255.0f*scale)));
	output[outY*mosaicW + outX] = pixel;
}
void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, const float scale, cudaStream_t stream){
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((inC + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, (W + blockDim.z - 1) / blockDim.z);
	FeatureMapMosaicKernel<<<gridDim, blockDim, 0, stream>>>(dInput, dOutput, H, W, inC, mosaicW, tileW, tileH, gridW, scale);
	checkCUDA(cudaGetLastError());
}
__global__ void GetPredictionKernel(const __half* predBatch, float* prediction, const int numCtrls, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < numCtrls){ prediction[idx] = __half2float(predBatch[idx + size - numCtrls]); }
}
void GetPrediction(const __half* predBatch, float* prediction, const int numCtrls, const int batchSize){
	float* devPtr = nullptr;
	cudaHostGetDevicePointer(&devPtr, prediction, 0);
	GetPredictionKernel<<<1, numCtrls>>>(predBatch, devPtr, numCtrls, batchSize*numCtrls);
	checkCUDA(cudaGetLastError());
	cudaDeviceSynchronize();
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
__global__ void AddBiasKernel(__half* output, const __half* bias, const int channels, const int batch){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = channels*batch;
	if(idx >= total){ return; }
	const int c = idx % channels;
	output[idx] = output[idx] + bias[c];
}
void AddBias(__half* output, const __half* bias, const int channels, const int batch){
	constexpr int bs = 256;
	const auto blocks = DivCeil(channels*batch, bs);
	AddBiasKernel<<<blocks, bs>>>(output, bias, channels, batch);
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
__global__ void TokensToSpatialKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	if(idx >= total) return;
	const int feature = idx % embedDim;
	const int tokenIndex = idx / embedDim % tokens;
	const int batchIndex = idx / (static_cast<size_t>(embedDim)*tokens);
	const int row = tokenIndex / patchCols;
	const int col = tokenIndex % patchCols;
	const size_t outIdx = ((static_cast<size_t>(batchIndex)*embedDim + feature)*patchRows + row)*patchCols + col;
	output[outIdx] = input[idx];
}
void TokensToSpatial(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	int bs = 256;
	const auto blocks = DivCeil(total, bs);
	TokensToSpatialKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}
__global__ void SpatialToTokensKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*embedDim*patchRows*patchCols;
	if(idx >= total) return;
	const int col = idx % patchCols;
	const int row = idx / patchCols % patchRows;
	const int feature = idx / (patchCols*patchRows) % embedDim;
	const int batchIndex = idx / (static_cast<size_t>(embedDim)*patchRows*patchCols);
	const int tokenIndex = row*patchCols + col;
	const size_t outIdx = (static_cast<size_t>(batchIndex)*tokens + tokenIndex)*embedDim + feature;
	output[outIdx] = input[idx];
}
void SpatialToTokens(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t total = static_cast<size_t>(batch)*embedDim*patchRows*patchCols;
	int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	SpatialToTokensKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}