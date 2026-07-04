#include "CuCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <climits>
#include <iostream>
constexpr int kCrossAttentionBlockSize = 256;
__device__ __forceinline__ float CrossAttentionLogit(const __half* __restrict__ Q, const __half* __restrict__ K, const int batch, const int head, const int query, const int key, const int queryTokens, const int contextTokens, const int headDim, const int heads, const float scale){
	const size_t qBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*headDim + static_cast<size_t>(query)*headDim;
	const size_t kBase = (static_cast<size_t>(batch)*heads + head)*contextTokens*headDim + static_cast<size_t>(key)*headDim;
	float sum = 0.0f;
	for(int d = 0; d < headDim; ++d){ sum = fmaf(__half2float(Q[qBase + d]), __half2float(K[kBase + d]), sum); }
	return sum*scale;
}
__device__ __forceinline__ float CrossAttentionMaskValue(const float* __restrict__ AttentionMask, const int batch, const int head, const int query, const int key, const int queryTokens, const int contextTokens, const int maskBatchSize, const int maskHeads){
	if(AttentionMask == nullptr) return 0.0f;
	const int effectiveMaskBatchSize = maskBatchSize > 0 ? maskBatchSize : 1;
	const int effectiveMaskHeads = maskHeads > 0 ? maskHeads : 1;
	const int maskBatch = batch % effectiveMaskBatchSize;
	const int maskHead = head % effectiveMaskHeads;
	const size_t maskIdx = (static_cast<size_t>(maskBatch)*effectiveMaskHeads + maskHead)*queryTokens*contextTokens + static_cast<size_t>(query)*contextTokens + key;
	return AttentionMask[maskIdx];
}
__global__ void CrossAttentionSoftmaxKernel(const __half* __restrict__ Q, const __half* __restrict__ K, float* __restrict__ AttentionWeights, const float* __restrict__ AttentionMask, const int batchSize, const int queryTokens, const int contextTokens, const int headDim, const int heads, const int maskBatchSize, const int maskHeads){
	__shared__ float shared[kCrossAttentionBlockSize];
	const int row = blockIdx.x;
	const int query = row % queryTokens;
	const int head = (row/queryTokens) % heads;
	const int batch = row/(queryTokens*heads);
	if(batch >= batchSize) return;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	const size_t attBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*contextTokens + static_cast<size_t>(query)*contextTokens;
	float localMax = -3.402823466e+38F;
	for(int key = threadIdx.x; key < contextTokens; key += blockDim.x){
		const float mask = CrossAttentionMaskValue(AttentionMask, batch, head, query, key, queryTokens, contextTokens, maskBatchSize, maskHeads);
		const float logit = CrossAttentionLogit(Q, K, batch, head, query, key, queryTokens, contextTokens, headDim, heads, scale) + mask;
		localMax = fmaxf(localMax, logit);
	}
	shared[threadIdx.x] = localMax;
	__syncthreads();
	for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(threadIdx.x < stride){ shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]); }
		__syncthreads();
	}
	const float rowMax = shared[0];
	float localSum = 0.0f;
	for(int key = threadIdx.x; key < contextTokens; key += blockDim.x){
		const float mask = CrossAttentionMaskValue(AttentionMask, batch, head, query, key, queryTokens, contextTokens, maskBatchSize, maskHeads);
		float value = 0.0f;
		if(mask > -1000.0f){
			const float logit = CrossAttentionLogit(Q, K, batch, head, query, key, queryTokens, contextTokens, headDim, heads, scale) + mask;
			const float shifted = logit - rowMax;
			if(shifted > -80.0f && shifted < 80.0f){ value = __expf(shifted); }
		}
		AttentionWeights[attBase + key] = value;
		localSum += value;
	}
	shared[threadIdx.x] = localSum;
	__syncthreads();
	for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(threadIdx.x < stride){ shared[threadIdx.x] += shared[threadIdx.x + stride]; }
		__syncthreads();
	}
	const float rowSum = shared[0];
	const float invSum = rowSum > 1e-20f ? 1.0f/rowSum : 0.0f;
	for(int key = threadIdx.x; key < contextTokens; key += blockDim.x){ AttentionWeights[attBase + key] *= invSum; }
}
__global__ void CrossAttentionWeightedValueKernel(const float* __restrict__ AttentionWeights, const __half* __restrict__ V, __half* __restrict__ Out, const int batchSize, const int queryTokens, const int contextTokens, const int headDim, const int heads){
	const size_t total = static_cast<size_t>(batchSize)*heads*queryTokens*headDim;
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x)*gridDim.x){
		const int d = static_cast<int>(idx % headDim);
		size_t tmp = idx/headDim;
		const int query = static_cast<int>(tmp % queryTokens);
		tmp /= queryTokens;
		const int head = static_cast<int>(tmp % heads);
		const int batch = static_cast<int>(tmp/heads);
		const size_t attBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*contextTokens + static_cast<size_t>(query)*contextTokens;
		const size_t vBase = (static_cast<size_t>(batch)*heads + head)*contextTokens*headDim + d;
		float sum = 0.0f;
		for(int key = 0; key < contextTokens; ++key){
			sum = fmaf(AttentionWeights[attBase + key], __half2float(V[vBase + static_cast<size_t>(key)*headDim]), sum);
		}
		Out[idx] = __float2half(sum);
	}
}
__global__ void CrossAttentionBackwardScoresKernel(const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ AttentionWeights, float* __restrict__ dScores, const int batchSize, const int queryTokens, const int contextTokens, const int headDim, const int heads){
	__shared__ float shared[kCrossAttentionBlockSize];
	const int row = blockIdx.x;
	const int query = row % queryTokens;
	const int head = (row/queryTokens) % heads;
	const int batch = row/(queryTokens*heads);
	if(batch >= batchSize) return;
	const size_t attBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*contextTokens + static_cast<size_t>(query)*contextTokens;
	const size_t outBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*headDim + static_cast<size_t>(query)*headDim;
	float localDot = 0.0f;
	for(int key = threadIdx.x; key < contextTokens; key += blockDim.x){
		const size_t vBase = (static_cast<size_t>(batch)*heads + head)*contextTokens*headDim + static_cast<size_t>(key)*headDim;
		float dProb = 0.0f;
		for(int d = 0; d < headDim; ++d){ dProb = fmaf(__half2float(dOut[outBase + d]), __half2float(V[vBase + d]), dProb); }
		localDot = fmaf(AttentionWeights[attBase + key], dProb, localDot);
	}
	shared[threadIdx.x] = localDot;
	__syncthreads();
	for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
		if(threadIdx.x < stride){ shared[threadIdx.x] += shared[threadIdx.x + stride]; }
		__syncthreads();
	}
	const float rowDot = shared[0];
	for(int key = threadIdx.x; key < contextTokens; key += blockDim.x){
		const size_t vBase = (static_cast<size_t>(batch)*heads + head)*contextTokens*headDim + static_cast<size_t>(key)*headDim;
		float dProb = 0.0f;
		for(int d = 0; d < headDim; ++d){ dProb = fmaf(__half2float(dOut[outBase + d]), __half2float(V[vBase + d]), dProb); }
		const float prob = AttentionWeights[attBase + key];
		dScores[attBase + key] = prob*(dProb - rowDot);
	}
}
__global__ void CrossAttentionBackwardQKernel(const float* __restrict__ dScores, const __half* __restrict__ K, __half* __restrict__ dQ, const int batchSize, const int queryTokens, const int contextTokens, const int headDim, const int heads){
	const size_t total = static_cast<size_t>(batchSize)*heads*queryTokens*headDim;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x)*gridDim.x){
		const int d = static_cast<int>(idx % headDim);
		size_t tmp = idx/headDim;
		const int query = static_cast<int>(tmp % queryTokens);
		tmp /= queryTokens;
		const int head = static_cast<int>(tmp % heads);
		const int batch = static_cast<int>(tmp/heads);
		const size_t scoreBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*contextTokens + static_cast<size_t>(query)*contextTokens;
		const size_t keyBase = (static_cast<size_t>(batch)*heads + head)*contextTokens*headDim + d;
		float sum = 0.0f;
		for(int key = 0; key < contextTokens; ++key){ sum = fmaf(dScores[scoreBase + key], __half2float(K[keyBase + static_cast<size_t>(key)*headDim]), sum); }
		dQ[idx] = __float2half(sum*scale);
	}
}
__global__ void CrossAttentionBackwardKKernel(const float* __restrict__ dScores, const __half* __restrict__ Q, __half* __restrict__ dK, const int batchSize, const int queryTokens, const int contextTokens, const int headDim, const int heads){
	const size_t total = static_cast<size_t>(batchSize)*heads*contextTokens*headDim;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x)*gridDim.x){
		const int d = static_cast<int>(idx % headDim);
		size_t tmp = idx/headDim;
		const int key = static_cast<int>(tmp % contextTokens);
		tmp /= contextTokens;
		const int head = static_cast<int>(tmp % heads);
		const int batch = static_cast<int>(tmp/heads);
		const size_t scoreBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*contextTokens + key;
		const size_t queryBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*headDim + d;
		float sum = 0.0f;
		for(int query = 0; query < queryTokens; ++query){ sum = fmaf(dScores[scoreBase + static_cast<size_t>(query)*contextTokens], __half2float(Q[queryBase + static_cast<size_t>(query)*headDim]), sum); }
		dK[idx] = __float2half(sum*scale);
	}
}
__global__ void CrossAttentionBackwardVKernel(const float* __restrict__ AttentionWeights, const __half* __restrict__ dOut, __half* __restrict__ dV, const int batchSize, const int queryTokens, const int contextTokens, const int headDim, const int heads){
	const size_t total = static_cast<size_t>(batchSize)*heads*contextTokens*headDim;
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x)*gridDim.x){
		const int d = static_cast<int>(idx % headDim);
		size_t tmp = idx/headDim;
		const int key = static_cast<int>(tmp % contextTokens);
		tmp /= contextTokens;
		const int head = static_cast<int>(tmp % heads);
		const int batch = static_cast<int>(tmp/heads);
		const size_t attBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*contextTokens + key;
		const size_t outBase = (static_cast<size_t>(batch)*heads + head)*queryTokens*headDim + d;
		float sum = 0.0f;
		for(int query = 0; query < queryTokens; ++query){ sum = fmaf(AttentionWeights[attBase + static_cast<size_t>(query)*contextTokens], __half2float(dOut[outBase + static_cast<size_t>(query)*headDim]), sum); }
		dV[idx] = __float2half(sum);
	}
}
bool CrossAttentionForward(const __half* Q, const __half* K, const __half* V, __half* Out, float* attentionWeights, const float* attentionMask, const int batchSize, const int queryTokens, const int contextTokens, const int headDim, const int heads, const int maskBatchSize, const int maskHeads){
	if(!Q || !K || !V || !Out || !attentionWeights){
		std::cerr << "CrossAttentionForward: Null pointer(s) provided" << std::endl;
		return false;
	}
	if(batchSize <= 0 || queryTokens <= 0 || contextTokens <= 0 || headDim <= 0 || heads <= 0){
		std::cerr << "CrossAttentionForward: Invalid dimensions" << std::endl;
		return false;
	}
	if(attentionMask != nullptr && (maskBatchSize <= 0 || maskHeads <= 0)){
		std::cerr << "CrossAttentionForward: Attention-mask dimensions must be positive" << std::endl;
		return false;
	}
	const size_t rowCount = static_cast<size_t>(batchSize)*heads*queryTokens;
	const size_t outputElements = rowCount*headDim;
	const size_t attentionElements = rowCount*contextTokens;
	if(rowCount == 0 || outputElements == 0 || attentionElements == 0 || rowCount > static_cast<size_t>(INT_MAX)){
		std::cerr << "CrossAttentionForward: Tensor is too large" << std::endl;
		return false;
	}
	CrossAttentionSoftmaxKernel<<<static_cast<unsigned int>(rowCount), kCrossAttentionBlockSize>>>(Q, K, attentionWeights, attentionMask, batchSize, queryTokens, contextTokens, headDim, heads, maskBatchSize, maskHeads);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		std::cerr << "CrossAttentionForward softmax error: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	size_t blocks = 0, tpb = 256;
	GetLaunchConfigGridStride(outputElements, blocks, tpb);
	CrossAttentionWeightedValueKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(attentionWeights, V, Out, batchSize, queryTokens, contextTokens, headDim, heads);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		std::cerr << "CrossAttentionForward value error: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	return true;
}
bool CrossAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* attentionWeights, __half* dQ, __half* dK, __half* dV, float* dScores, const int batchSize, const int queryTokens, const int contextTokens, const int headDim, const int heads){
	if(!Q || !K || !V || !dOut || !attentionWeights || !dQ || !dK || !dV || !dScores){
		std::cerr << "CrossAttentionBackward: Null pointer(s) provided" << std::endl;
		return false;
	}
	if(batchSize <= 0 || queryTokens <= 0 || contextTokens <= 0 || headDim <= 0 || heads <= 0){
		std::cerr << "CrossAttentionBackward: Invalid dimensions" << std::endl;
		return false;
	}
	const size_t rowCount = static_cast<size_t>(batchSize)*heads*queryTokens;
	const size_t queryElements = rowCount*headDim;
	const size_t contextElements = static_cast<size_t>(batchSize)*heads*contextTokens*headDim;
	const size_t attentionElements = rowCount*contextTokens;
	if(rowCount == 0 || queryElements == 0 || contextElements == 0 || attentionElements == 0 || rowCount > static_cast<size_t>(INT_MAX)){
		std::cerr << "CrossAttentionBackward: Tensor is too large" << std::endl;
		return false;
	}
	CrossAttentionBackwardScoresKernel<<<static_cast<unsigned int>(rowCount), kCrossAttentionBlockSize>>>(V, dOut, attentionWeights, dScores, batchSize, queryTokens, contextTokens, headDim, heads);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		std::cerr << "CrossAttentionBackward scores error: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	size_t blocks = 0, tpb = 256;
	GetLaunchConfigGridStride(queryElements, blocks, tpb);
	CrossAttentionBackwardQKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(dScores, K, dQ, batchSize, queryTokens, contextTokens, headDim, heads);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		std::cerr << "CrossAttentionBackward dQ error: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	GetLaunchConfigGridStride(contextElements, blocks, tpb);
	CrossAttentionBackwardKKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(dScores, Q, dK, batchSize, queryTokens, contextTokens, headDim, heads);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		std::cerr << "CrossAttentionBackward dK error: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	CrossAttentionBackwardVKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(attentionWeights, dOut, dV, batchSize, queryTokens, contextTokens, headDim, heads);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		std::cerr << "CrossAttentionBackward dV error: " << cudaGetErrorString(err) << std::endl;
		return false;
	}
	return true;
}
