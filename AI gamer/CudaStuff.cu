#include "CuCommon.cuh"
#include <cuda.h>
#include <curand.h>
void BlockShiftHalf(__half* hPtr, const int shiftBy, const int blocksToShift){
	auto blockSize = shiftBy;
	if(blockSize < 0) blockSize = -blockSize;
	if(shiftBy > 0){ for(int i = blocksToShift; 0 < i; --i){ cudaMemcpy(hPtr + i * blockSize, hPtr + (i - 1) * blockSize, blockSize * sizeof(__half), cudaMemcpyDeviceToDevice); } } else{
		for(int i = 0; i < blocksToShift; ++i){ cudaMemcpy(hPtr + (i - 1) * blockSize, hPtr + i * blockSize, blockSize * sizeof(__half), cudaMemcpyDeviceToDevice); }
	}
}
__global__ void GradientKernel(__half* grads, const __half* predictions, const __half* targets, const float clip, const int size){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){ grads[idx] = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx] - targets[idx])))); }
}
void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const float clip, const int size){
	auto gridSize = DivCeil(size, BS);
	GradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, size);
}
__global__ void SplitGradKernel(__half* gradients, const __half* predictions, const float* targets, const float clip, const int numCtrls, const int numButs, const int batchSize, const int size){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx / numCtrls;
		const int ctrlId = idx % numCtrls;
		const auto diff = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx]) - targets[idx])));
		if(ctrlId < numButs){
			const auto gradIdx = batchId * numButs + ctrlId;
			gradients[gradIdx] = diff;
		} else{
			const auto gradIdx = numButs * batchSize + batchId * (numCtrls - numButs) + (ctrlId - numButs);
			gradients[gradIdx] = diff;
		}
	}
}
void SplitGradient(__half* dGradient, const __half* dPredictions, const float* dTargets, const float clip, const int size, const int numCtrls, const int numButs, const int batchSize){
	auto gridSize = DivCeil(size, BS);
	SplitGradKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, numCtrls, numButs, batchSize, size);
}
__global__ void MergeOutputsKernel(__half* predOut, const __half* buttonData, const __half* axisData, const int size, const int numCtrls, const int numButs){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx / numCtrls;
		const int ctrlId = idx % numCtrls;
		if(ctrlId < numButs){ predOut[idx] = buttonData[batchId * numButs + ctrlId]; } else{ predOut[idx] = axisData[batchId * (numCtrls - numButs) + (ctrlId - numButs)]; }
	}
}
void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, const int numCtrls, const int numButs, const int size){
	auto gridSize = DivCeil(size, BS);
	MergeOutputsKernel<<<gridSize, BS>>>(predOut, buttonData, axisData, size, numCtrls, numButs);
}
__global__ void BCEGradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size, const float scale){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		const float y = __half2float(targets[idx]);
		const float pClamped = fminf(fmaxf(__half2float(predictions[idx]), EPSILON_F), 1.0f - EPSILON_F);
		float gradient = 0.0f;
		if(y == 1.0f){
			gradient = (pClamped - 1.0f) / pClamped;
			gradients[idx] = __float2half(gradient * scale);
		} else{
			gradient = pClamped / (1.0f - pClamped);
			gradients[idx] = __float2half(gradient * scale);
		}
	}
}
void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size, const float scale){
	auto gridSize = DivCeil(size, BS);
	BCEGradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, scale);
}
__device__ int deviceResult;
__global__ void isNaNKernel(const __half* __restrict__ data, int size){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size && __hisnan(data[idx])){ atomicExch(&deviceResult, 1); }
}
bool IsnanHalf(const __half* __restrict__ data, int size){
	int hResult = 0;
	cudaMemcpyToSymbol(deviceResult, &hResult, sizeof(int));
	auto gridSize = DivCeil(size, BS);
	isNaNKernel<<<gridSize, BS>>>(data, size);
	cudaMemcpyFromSymbol(&hResult, deviceResult, sizeof(int));
	return hResult != 0;
}
__global__ void FeatureMapMosaicKernel(const __half* __restrict__ input, unsigned char* __restrict__ output, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW){
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.z * blockDim.z + threadIdx.z;
	if(c >= inC || y >= H || x >= W) return;
	const int tileX = c % gridW;
	const int tileY = c / gridW;
	const int outX = tileX * tileW + x;
	const int outY = tileY * tileH + y;
	const __half value = input[c * H * W + y * W + x];
	const float fVal = __half2float(value);
	const unsigned char pixel = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, fVal * 255.0f)));
	output[outY * mosaicW + outX] = pixel;
}
void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, cudaStream_t stream){
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((inC + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, (W + blockDim.z - 1) / blockDim.z);
	FeatureMapMosaicKernel<<<gridDim, blockDim, 0, stream>>>(dInput, dOutput, H, W, inC, mosaicW, tileW, tileH, gridW);
}
__global__ void GetPredictionKernel(const __half* predBatch, float* prediction, const int numCtrls, const int size){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numCtrls){ prediction[idx] = __half2float(predBatch[idx + size - numCtrls]); }
}
void GetPrediction(const __half* predBatch, float* prediction, const int numCtrls, const int batchSize){
	float* devPtr = nullptr;
	cudaHostGetDevicePointer(&devPtr, prediction, 0);
	GetPredictionKernel<<<1, numCtrls>>>(predBatch, devPtr, numCtrls, batchSize * numCtrls);
	cudaDeviceSynchronize();
}
__global__ void ExtractPatchesKernel(const __half* __restrict__ x, __half* __restrict__ y, int B, int C, int H, int W, int P, bool zeroPad){
	const int kPatchArea = P * P;
	const int PH = (H + P - 1) / P;
	const int PW = (W + P - 1) / P;
	const long patchDim = static_cast<long>(C) * kPatchArea;
	const long total = static_cast<long>(B) * PH * PW * patchDim;
	long idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= total) return;
	long patch = idx / patchDim;
	int inPatch = idx - patch * patchDim;
	int b = patch / (PH * PW);
	int pp = patch - b * (PH * PW);
	int pyPatch = pp / PW;
	int pxPatch = pp - pyPatch * PW;
	int c = inPatch / kPatchArea;
	int rem = inPatch - c * kPatchArea;
	int py = rem / P;
	int px = rem - py * P;
	int srcY = pyPatch * P + py;
	int srcX = pxPatch * P + px;
	__half val = __float2half(0.f);
	if(srcY < H && srcX < W){
		long srcIdx = (((static_cast<long>(b) * C + c) * H + srcY) * W) + srcX;
		val = x[srcIdx];
	} else if(!zeroPad){ return; }
	y[idx] = val;
}
void ExtractPatches(const __half* in, __half* out, int B, int C, int H, int W, int P, bool zeroPad){
	const int PH = (H + P - 1) / P;
	const int PW = (W + P - 1) / P;
	const size_t total = static_cast<size_t>(B) * PH * PW * C * P * P;
	int blocks, tpb;
	GetLaunchConfig(total, blocks, tpb);
	ExtractPatchesKernel<<<blocks, tpb>>>(in, out, B, C, H, W, P, zeroPad);
}
__global__ void CombinePatchGradsKernel(const __half* __restrict__ dy, __half* __restrict__ dx, int B, int C, int H, int W, int P){
	const int kPatchArea = P * P;
	const int PH = (H + P - 1) / P;
	const int PW = (W + P - 1) / P;
	const long patchDim = static_cast<long>(C) * kPatchArea;
	const long total = static_cast<long>(B) * PH * PW * patchDim;
	long idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= total) return;
	long patch = idx / patchDim;
	int inPatch = idx - patch * patchDim;
	int b = patch / (PH * PW);
	int pp = patch - b * (PH * PW);
	int pyPatch = pp / PW;
	int pxPatch = pp - pyPatch * PW;
	int c = inPatch / kPatchArea;
	int rem = inPatch - c * kPatchArea;
	int py = rem / P;
	int px = rem - py * P;
	int dstY = pyPatch * P + py;
	int dstX = pxPatch * P + px;
	if(dstY >= H || dstX >= W) return;   
	long dstIdx = (((static_cast<long>(b) * C + c) * H + dstY) * W) + dstX;
	dx[dstIdx] = dy[idx];
}
void CombinePatchGrads(const __half* dy, __half* dx, int B, int C, int H, int W, int P){
	const int PH = (H + P - 1) / P;
	const int PW = (W + P - 1) / P;
	const size_t total = static_cast<size_t>(B)*PH*PW*C*P*P;
	int blocks, tpb;
	GetLaunchConfig(total, blocks, tpb);
	CombinePatchGradsKernel<<<blocks, tpb>>>(dy, dx, B, C, H, W, P);
}