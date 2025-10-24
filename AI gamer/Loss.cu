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
__global__ void MseLoss2Kernel(const __half* predictions, const float* targets, const int size, const int numKeys, const int numCtrls){
	extern __shared__ float sdata[];
	const int tid = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float sumKeys = 0.0f;
	float sumMouse = 0.0f;
	while(idx < size){
		const float pred = __half2float(predictions[idx]);
		const float target = targets[idx];
		const bool isKey = idx % numCtrls < numKeys;
		if(isKey){
			sumKeys += BceWithLogitsLoss(pred, target);
		} else{
			const float diff = pred - target;
			sumMouse += diff*diff;
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
void MseLoss2(const __half* dPredictions, const float* dTargets, const int numButs, const int numCtrls, const int batchSize, float* butLoss, float* axesLoss){
	constexpr auto zero = 0.0f;
	const auto size = numCtrls*batchSize;
	cudaMemcpyToSymbol(dLossKeys, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dLossMouse, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	auto gridSize = DivCeil(size, BS);
	MseLoss2Kernel<<<gridSize, BS, 2*BS*sizeof(float)>>>(dPredictions, dTargets, size, numButs, numCtrls);
	cudaMemcpyFromSymbol(butLoss, dLossKeys, sizeof(float));
	cudaMemcpyFromSymbol(axesLoss, dLossMouse, sizeof(float));
	*butLoss /= numButs*batchSize;
	*axesLoss /= (numCtrls - numButs)*batchSize;
}