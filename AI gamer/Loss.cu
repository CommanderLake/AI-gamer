#define __CUDACC__
#include "CuCommon.cuh"
#include <device_launch_parameters.h>
#include <device_functions.h>
__device__ float dLoss;
__global__ void MseLossKernel(const __half* predictions, const float* targets, int size){
	extern __shared__ float sdata[];
	const int tid = threadIdx.x;
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float diff = 0.0f;
	if(idx < size){
		diff = __half2float(predictions[idx]) - targets[idx];
		diff *= diff;
	}
	sdata[tid] = diff;
	__syncthreads();
	for(int i = blockDim.x / 2; i > 0; i >>= 1){
		if(tid < i){ sdata[tid] += sdata[tid + i]; }
		__syncthreads();
	}
	if(tid == 0){ atomicAdd(&dLoss, sdata[0]); }
}
float MseLoss(const __half* dPredictions, const float* dTargets, int size){
	constexpr auto zero = 0.0f;
	cudaMemcpyToSymbol(dLoss, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	auto gridSize = DivCeil(size, BS);
	MseLossKernel<<<gridSize, BS, BS*sizeof(float)>>>(dPredictions, dTargets, size);
	checkCUDA(cudaGetLastError());
	float h_loss;
	cudaMemcpyFromSymbol(&h_loss, dLoss, sizeof(float));
	return h_loss / size;
}
__device__ float dLossKeys;
__device__ float dLossMouse;
__device__ inline float BceWithLogitsLoss(const float logit, const float target){
	const float maxPart = fmaxf(logit, 0.0f);
	const float negAbs = -fabsf(logit);
	return maxPart - logit*target + log1pf(expf(negAbs));
}
__global__ void LossStatsKernel(const __half* predictions, const float* targets, const int size, const int numKeys, const int numCtrls){
	extern __shared__ float sdata[];
	const int tid = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float sumKeys = 0.0f;
	float sumMouse = 0.0f;
	const int numAxisOutputs = numCtrls - numKeys;
	const int numAxes = numAxisOutputs / 2;
	constexpr float kLogSigmaMin = -5.0f;
	constexpr float kLogSigmaMax = 2.0f;
	constexpr float kLogSigmaL2 = 0.01f;
	while(idx < size){
		const float pred = __half2float(predictions[idx]);
		const float target = targets[idx];
		const bool isKey = idx % numCtrls < numKeys;
		if(isKey){
			sumKeys += BceWithLogitsLoss(pred, target);
		} else{
			const int batchId = idx / numCtrls;
			const int axisOffset = idx - (batchId*numCtrls + numKeys);
			if(axisOffset < numAxes){
				const float mu = pred;
				float logSigma = __half2float(predictions[idx + numAxes]);
				logSigma = fmaxf(kLogSigmaMin, fminf(kLogSigmaMax, logSigma));
				const float diff = mu - target;
				const float invVar = expf(-2.0f*logSigma);
				sumMouse += 0.5f*diff*diff*invVar + logSigma + kLogSigmaL2*logSigma*logSigma;
			}
		}
		idx += gridDim.x*blockDim.x;
	}
	sdata[tid] = sumKeys;
	sdata[tid + blockDim.x] = sumMouse;
	__syncthreads();
	for(int s = blockDim.x / 2; s > 0; s >>= 1){
		if(tid < s){
			sdata[tid] += sdata[tid + s];
			sdata[tid + blockDim.x] += sdata[tid + blockDim.x + s];
		}
		__syncthreads();
	}
	if(tid == 0){
		atomicAdd(&dLossKeys, sdata[0]);
		atomicAdd(&dLossMouse, sdata[blockDim.x]);
	}
}
void LossStats(const __half* dPredictions, const float* dTargets, const int numButs, const int numCtrls, const int batchSize, float* butLoss, float* axesLoss){
	constexpr auto zero = 0.0f;
	const auto size = numCtrls*batchSize;
	const auto numAxes = (numCtrls - numButs) / 2;
	cudaMemcpyToSymbol(dLossKeys, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dLossMouse, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	auto gridSize = DivCeil(size, BS);
	LossStatsKernel<<<gridSize, BS, 2*BS*sizeof(float)>>>(dPredictions, dTargets, size, numButs, numCtrls);
	checkCUDA(cudaGetLastError());
	cudaMemcpyFromSymbol(butLoss, dLossKeys, sizeof(float));
	cudaMemcpyFromSymbol(axesLoss, dLossMouse, sizeof(float));
	*butLoss /= numButs*batchSize;
	*axesLoss /= numAxes*batchSize;
}