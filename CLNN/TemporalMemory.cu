#define __CUDACC__
#include "CuCommon.cuh"
#include <cuda.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cfloat>
namespace{
__global__ void TemporalScoresKernel(const __half* query, const __half* keys, const int* validCounts, float* attnWeights, int batchSize, int maxContext, int numHeads, int headDim, float invSqrtDim){
	const int batchHead = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalBatchHeads = batchSize * numHeads;
	if(batchHead >= totalBatchHeads){ return; }
	const int b = batchHead / numHeads;
	const int h = batchHead % numHeads;
	const int valid = max(0, min(validCounts[b], maxContext));
	const __half* qPtr = query + (static_cast<size_t>(b) * numHeads + h) * headDim;
	float maxScore = -FLT_MAX;
	for(int t = 0; t < valid; ++t){
		const __half* kPtr = keys + ((static_cast<size_t>(b) * maxContext + t) * numHeads + h) * headDim;
		float score = 0.0f;
		for(int d = 0; d < headDim; ++d){ score = fmaf(__half2float(qPtr[d]), __half2float(kPtr[d]), score); }
		score *= invSqrtDim;
		attnWeights[(static_cast<size_t>(b) * numHeads + h) * maxContext + t] = score;
		maxScore = fmaxf(maxScore, score);
	}
	float sum = 0.0f;
	for(int t = 0; t < valid; ++t){
		const size_t idx = (static_cast<size_t>(b) * numHeads + h) * maxContext + t;
		const float expVal = expf(attnWeights[idx] - maxScore);
		attnWeights[idx] = expVal;
		sum += expVal;
	}
	const float invSum = sum > 0.0f ? 1.0f / sum : 0.0f;
	for(int t = 0; t < valid; ++t){
		const size_t idx = (static_cast<size_t>(b) * numHeads + h) * maxContext + t;
		attnWeights[idx] *= invSum;
	}
	for(int t = valid; t < maxContext; ++t){ attnWeights[(static_cast<size_t>(b) * numHeads + h) * maxContext + t] = 0.0f; }
}
__global__ void TemporalWeightedSumKernel(const __half* values, const float* attnWeights, const int* validCounts, __half* output, int batchSize, int maxContext, int numHeads, int headDim){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = batchSize * numHeads * headDim;
	if(idx >= total){ return; }
	const int d = idx % headDim;
	const int h = idx / headDim % numHeads;
	const int b = idx / (headDim * numHeads);
	const int valid = max(0, min(validCounts[b], maxContext));
	float sum = 0.0f;
	for(int t = 0; t < valid; ++t){
		const float weight = attnWeights[(static_cast<size_t>(b) * numHeads + h) * maxContext + t];
		const __half value = values[((static_cast<size_t>(b) * maxContext + t) * numHeads + h) * headDim + d];
		sum = fmaf(weight, __half2float(value), sum);
	}
	output[idx] = __float2half(sum);
}
__global__ void TemporalBackwardKernel(const __half* query, const __half* keys, const __half* values, const __half* gradOut, const float* attnWeights, const int* validCounts, __half* gradQuery, __half* gradKeys, __half* gradValues, float* gradScores, int batchSize, int maxContext, int numHeads, int headDim, float invSqrtDim){
	const int batchHead = blockIdx.x * blockDim.x + threadIdx.x;
	const int totalBatchHeads = batchSize * numHeads;
	if(batchHead >= totalBatchHeads){ return; }
	const int b = batchHead / numHeads;
	const int h = batchHead % numHeads;
	const int valid = max(0, min(validCounts[b], maxContext));
	const __half* qPtr = query + (static_cast<size_t>(b) * numHeads + h) * headDim;
	const __half* gradPtr = gradOut + (static_cast<size_t>(b) * numHeads + h) * headDim;
	float attnGradDotSum = 0.0f;
	for(int t = 0; t < valid; ++t){
		const __half* vPtr = values + ((static_cast<size_t>(b) * maxContext + t) * numHeads + h) * headDim;
		float dot = 0.0f;
		for(int d = 0; d < headDim; ++d){ dot = fmaf(__half2float(gradPtr[d]), __half2float(vPtr[d]), dot); }
		const size_t scoreIdx = (static_cast<size_t>(b) * numHeads + h) * maxContext + t;
		gradScores[scoreIdx] = dot;
		attnGradDotSum += dot * attnWeights[scoreIdx];
	}
	for(int d = 0; d < headDim; ++d){
		float gq = 0.0f;
		for(int t = 0; t < valid; ++t){
			const size_t scoreIdx = (static_cast<size_t>(b) * numHeads + h) * maxContext + t;
			const float gScore = attnWeights[scoreIdx] * (gradScores[scoreIdx] - attnGradDotSum);
			const __half* kPtr = keys + ((static_cast<size_t>(b) * maxContext + t) * numHeads + h) * headDim;
			const size_t vkIdx = ((static_cast<size_t>(b) * maxContext + t) * numHeads + h) * headDim + d;
			const float gradValue = attnWeights[scoreIdx] * __half2float(gradPtr[d]);
			gradValues[vkIdx] = __float2half(gradValue);
			gradKeys[vkIdx] = __float2half(gScore * __half2float(qPtr[d]) * invSqrtDim);
			gq = fmaf(gScore * invSqrtDim, __half2float(kPtr[d]), gq);
		}
		gradQuery[(static_cast<size_t>(b) * numHeads + h) * headDim + d] = __float2half(gq);
	}
	for(int t = valid; t < maxContext; ++t){
		for(int d = 0; d < headDim; ++d){
			const size_t idx = ((static_cast<size_t>(b) * maxContext + t) * numHeads + h) * headDim + d;
			gradValues[idx] = __float2half(0.0f);
			gradKeys[idx] = __float2half(0.0f);
		}
		gradScores[(static_cast<size_t>(b) * numHeads + h) * maxContext + t] = 0.0f;
	}
}
}
void TemporalMemoryForward(const __half* query, const __half* keys, const __half* values, const int* validCounts, float* attnWeights, __half* output, const int batchSize, const int maxContext, const int numHeads, const int headDim){
	const float invSqrtDim = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	const int totalBatchHeads = batchSize * numHeads;
	const int threads = 128;
	const int blocks = DivCeil(totalBatchHeads, threads);
	TemporalScoresKernel<<<blocks, threads>>>(query, keys, validCounts, attnWeights, batchSize, maxContext, numHeads, headDim, invSqrtDim);
	checkCUDA(cudaGetLastError());
	const int totalOut = batchSize * numHeads * headDim;
	const int outBlocks = DivCeil(totalOut, threads);
	TemporalWeightedSumKernel<<<outBlocks, threads>>>(values, attnWeights, validCounts, output, batchSize, maxContext, numHeads, headDim);
	checkCUDA(cudaGetLastError());
}
void TemporalMemoryBackward(const __half* query, const __half* keys, const __half* values, const __half* gradOut, const float* attnWeights, const int* validCounts, __half* gradQuery, __half* gradKeys, __half* gradValues, float* gradScores, const int batchSize, const int maxContext, const int numHeads, const int headDim){
	const float invSqrtDim = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	const int totalBatchHeads = batchSize * numHeads;
	const int threads = 128;
	const int blocks = DivCeil(totalBatchHeads, threads);
	TemporalBackwardKernel<<<blocks, threads>>>(query, keys, values, gradOut, attnWeights, validCounts, gradQuery, gradKeys, gradValues, gradScores, batchSize, maxContext, numHeads, headDim, invSqrtDim);
	checkCUDA(cudaGetLastError());
}
