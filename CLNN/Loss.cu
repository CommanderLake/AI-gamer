#define __CUDACC__
#include "CuCommon.cuh"
#include <device_launch_parameters.h>
#include <device_functions.h>
__device__ float dLossKeys;
__device__ float dLossMouse;
__device__ inline float BceWithLogitsLoss(const float logit, const float target){
	const float maxPart = fmaxf(logit, 0.0f);
	const float negAbs = -fabsf(logit);
	return maxPart - logit*target + log1pf(expf(negAbs));
}
__device__ inline float AxisLossWeight(const float target){
	const float w = 1.0f + 0.5f*fabsf(target);
	return fminf(w, 4.0f);
}
__device__ inline float AxisSmoothL1Loss(const float diff){
	constexpr float beta = 1.0f;
	const float absDiff = fabsf(diff);
	if(absDiff < beta){ return 0.5f*diff*diff/beta; }
	return absDiff - 0.5f*beta;
}
__device__ inline float AxisSmoothL1Grad(const float diff){
	constexpr float beta = 1.0f;
	const float absDiff = fabsf(diff);
	if(absDiff < beta){ return diff/beta; }
	return diff >= 0.0f ? 1.0f : -1.0f;
}
__global__ void LossStatsKernel(const __half* predictions, const float* targets, const int size, const int numKeys, const int numCtrls){
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
			const float w = AxisLossWeight(target);
			sumMouse += w*AxisSmoothL1Loss(diff);
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
	cudaMemcpyToSymbol(dLossKeys, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(dLossMouse, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	auto gridSize = DivCeil(size, BS);
	LossStatsKernel<<<gridSize, BS, 2*BS*sizeof(float)>>>(dPredictions, dTargets, size, numButs, numCtrls);
	checkCUDA(cudaGetLastError());
	cudaMemcpyFromSymbol(butLoss, dLossKeys, sizeof(float));
	cudaMemcpyFromSymbol(axesLoss, dLossMouse, sizeof(float));
	*butLoss /= numButs*batchSize;
	*axesLoss /= (numCtrls - numButs)*batchSize;
}
__device__ inline float Sigmoidf(const float x){
	if(x >= 0.0f){
		const float z = __expf(-x);
		return 1.0f/(1.0f + z);
	}
	const float z = __expf(x);
	return z/(1.0f + z);
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
			const float w = AxisLossWeight(target);
			gradients[numButs*batchSize + batchId*(numCtrls - numButs) + (ctrlId - numButs)] = __float2half(fmaxf(-clip, fminf(clip, w*AxisSmoothL1Grad(pred - target))));
		}
	}
}
void LossBackprop(__half* dGradient, const __half* dPredictions, const float* dTargets, const float clip, const int size, const int numCtrls, const int numButs, const int batchSize){
	auto gridSize = DivCeil(size, BS);
	LossBackpropKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, numCtrls, numButs, batchSize, size);
	checkCUDA(cudaGetLastError());
}