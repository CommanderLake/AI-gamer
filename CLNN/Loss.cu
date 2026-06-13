#define __CUDACC__
#include "CuCommon.h"
#include <device_launch_parameters.h>
#include <device_functions.h>
__device__ float dLossKeys;
__device__ float dLossMouse;
__device__ float dLossSmoothL1;
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
			const float w = AxisLossWeight(target);
			sumMouse += w*AxisSmoothL1Loss(pred - target);
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
	auto gridSize = DivCeil(size, 256);
	LossStatsKernel<<<gridSize, 256, 2*256*sizeof(float)>>>(dPredictions, dTargets, size, numButs, numCtrls);
	checkCUDA(cudaGetLastError());
	cudaMemcpyFromSymbol(butLoss, dLossKeys, sizeof(float));
	cudaMemcpyFromSymbol(axesLoss, dLossMouse, sizeof(float));
	*butLoss /= numButs*batchSize;
	*axesLoss /= (numCtrls - numButs)*batchSize;
}
__device__ inline float Sigmoidf(const float x){
	if(x >= 0.0f){
		return 1.0f/(1.0f + __expf(-x));
	}
	const float z = __expf(x);
	return z/(1.0f + z);
}
__global__ void LossBackpropKernel(__half* gradients, const __half* predictions, const float* targets, const float clip, const int numCtrls, const int numButs, const int batchSize, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const int ctrlId = idx % numCtrls;
		const float target = targets[idx];
		if(ctrlId < numButs){
			const float pred = Sigmoidf(__half2float(predictions[idx]));
			gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, pred - target)));
		} else{
			const float pred = __half2float(predictions[idx]);
			const float w = AxisLossWeight(target);
			gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, w*AxisSmoothL1Grad(pred - target))));
		}
	}
}
void LossBackprop(__half* dGradient, const __half* dPredictions, const float* dTargets, const float clip, const int size, const int numCtrls, const int numButs, const int batchSize){
	auto gridSize = DivCeil(size, 256);
	LossBackpropKernel<<<gridSize, 256>>>(dGradient, dPredictions, dTargets, clip, numCtrls, numButs, batchSize, size);
	checkCUDA(cudaGetLastError());
}
__device__ inline float SmoothL1Loss(const float diff, const float beta){
	const float absDiff = fabsf(diff);
	if(beta <= 0.0f){ return absDiff; }
	if(absDiff < beta){ return 0.5f*diff*diff/beta; }
	return absDiff - 0.5f*beta;
}
__device__ inline float SmoothL1Grad(const float diff, const float beta){
	if(beta <= 0.0f){ return diff >= 0.0f ? 1.0f : -1.0f; }
	const float absDiff = fabsf(diff);
	if(absDiff < beta){ return diff/beta; }
	return diff >= 0.0f ? 1.0f : -1.0f;
}
__global__ void SmoothL1LossStatsKernel(const __half* predictions, const __half* targets, const float beta, const int size){
	extern __shared__ float sdata[];
	const int tid = threadIdx.x;
	const int stride = gridDim.x*blockDim.x;
	float sum = 0.0f;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < size; idx += stride){
		const float diff = __half2float(predictions[idx]) - __half2float(targets[idx]);
		sum += SmoothL1Loss(diff, beta);
	}
	sdata[tid] = sum;
	__syncthreads();
	for(int s = blockDim.x/2; s > 0; s >>= 1){
		if(tid < s){ sdata[tid] += sdata[tid + s]; }
		__syncthreads();
	}
	if(tid == 0){ atomicAdd(&dLossSmoothL1, sdata[0]); }
}
void SmoothL1LossStats(const __half* dPredictions, const __half* dTargets, const float beta, const int size, float* loss){
	if(size <= 0){
		*loss = 0.0f;
		return;
	}
	constexpr auto zero = 0.0f;
	cudaMemcpyToSymbol(dLossSmoothL1, &zero, sizeof(float), 0, cudaMemcpyHostToDevice);
	auto gridSize = DivCeil(size, 256);
	SmoothL1LossStatsKernel<<<gridSize, 256, 256*sizeof(float)>>>(dPredictions, dTargets, beta, size);
	checkCUDA(cudaGetLastError());
	cudaMemcpyFromSymbol(loss, dLossSmoothL1, sizeof(float));
	*loss /= size;
}
__global__ void SmoothL1LossBackpropKernel(__half* gradients, const __half* predictions, const __half* targets, const float beta, const float clip, const int size, const float gradientScale){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const float diff = __half2float(predictions[idx]) - __half2float(targets[idx]);
		const float grad = SmoothL1Grad(diff, beta)*gradientScale;
		gradients[idx] = __float2half(fmaxf(-clip, fminf(clip, grad)));
	}
}
void SmoothL1LossBackprop(__half* dGradient, const __half* dPredictions, const __half* dTargets, const float beta, const float clip, const int size, const float gradientScale){
	if(size <= 0) return;
	auto gridSize = DivCeil(size, 256);
	SmoothL1LossBackpropKernel<<<gridSize, 256>>>(dGradient, dPredictions, dTargets, beta, clip, size, gradientScale);
	checkCUDA(cudaGetLastError());
}
