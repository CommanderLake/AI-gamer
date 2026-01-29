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
__global__ void LossStatsKernel(const __half* predictions, const float* targets, const int size, const int numKeys, const int numCtrls){
	extern __shared__ float sdata[];
	const int tid = threadIdx.x;
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	float sumKeys = 0.0f;
	float sumMouse = 0.0f;
	const int numAxisOutputs = numCtrls - numKeys;
	const int numAxes = numAxisOutputs/2;
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
			const int batchId = idx/numCtrls;
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
	for(int s = blockDim.x/2; s > 0; s >>= 1){
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
	const auto numAxes = (numCtrls - numButs)/2;
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
		const int batchId = idx/numCtrls;
		const int ctrlId = idx % numCtrls;
		const float target = targets[idx];
		if(ctrlId < numButs){
			const float logit = __half2float(predictions[idx]);
			const float prob = Sigmoidf(logit);
			gradients[batchId*numButs + ctrlId] = __float2half(fmaxf(-clip, fminf(clip, prob - target)));
		} else{
			const int axisId = ctrlId - numButs;
			const int numAxisOutputs = numCtrls - numButs;
			const int numAxes = numAxisOutputs/2;
			constexpr float kLogSigmaMin = -5.0f;
			constexpr float kLogSigmaMax = 2.0f;
			constexpr float kLogSigmaL2 = 0.01f;
			if(axisId < numAxes){
				const float mu = __half2float(predictions[idx]);
				float logSigma = __half2float(predictions[idx + numAxes]);
				logSigma = fmaxf(kLogSigmaMin, fminf(kLogSigmaMax, logSigma));
				const float diff = mu - target;
				const float invVar = expf(-2.0f*logSigma);
				const float gradMu = fmaxf(-clip, fminf(clip, diff*invVar));
				const float gradLogSigmaRaw = 1.0f - diff*diff*invVar + 2.0f*kLogSigmaL2*logSigma;
				const float gradLogSigma = fmaxf(-clip, fminf(clip, gradLogSigmaRaw));
				const int axisBase = numButs*batchSize + batchId*numAxisOutputs;
				gradients[axisBase + axisId] = __float2half(gradMu);
				gradients[axisBase + axisId + numAxes] = __float2half(gradLogSigma);
			}
		}
	}
}
void LossBackprop(__half* dGradient, const __half* dPredictions, const float* dTargets, const float clip, const int size, const int numCtrls, const int numButs, const int batchSize){
	auto gridSize = DivCeil(size, BS);
	LossBackpropKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, numCtrls, numButs, batchSize, size);
	checkCUDA(cudaGetLastError());
}