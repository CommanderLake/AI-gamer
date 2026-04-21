#define __CUDACC__
#include "CuCommon.h"
#include <cuda.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
__forceinline__ __device__ float WarpReduceSum(float val){
	const unsigned mask = __activemask();
	for(int offset = warpSize >> 1; offset > 0; offset >>= 1){ val += __shfl_down_sync(mask, val, offset); }
	return val;
}
__forceinline__ __device__ float WarpReduceMax(float val){
	const unsigned mask = __activemask();
	for(int offset = warpSize >> 1; offset > 0; offset >>= 1){ val = fmaxf(val, __shfl_down_sync(mask, val, offset)); }
	return val;
}
__forceinline__ __device__ float BlockReduceSum(float val){
	__shared__ float shared[32];
	const int lane = threadIdx.x & (warpSize - 1);
	const int wid = threadIdx.x >> 5;
	val = WarpReduceSum(val);
	if(lane == 0){ shared[wid] = val; }
	__syncthreads();
	const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
	float blockVal = 0.0f;
	if(wid == 0){
		blockVal = lane < numWarps ? shared[lane] : 0.0f;
		blockVal = WarpReduceSum(blockVal);
		if(lane == 0){ shared[0] = blockVal; }
	}
	__syncthreads();
	return shared[0];
}
__forceinline__ __device__ float BlockReduceMax(float val){
	__shared__ float shared[32];
	const int lane = threadIdx.x & (warpSize - 1);
	const int wid = threadIdx.x >> 5;
	val = WarpReduceMax(val);
	if(lane == 0){ shared[wid] = val; }
	__syncthreads();
	const int numWarps = (blockDim.x + warpSize - 1) / warpSize;
	float blockVal = -FLT_MAX;
	if(wid == 0){
		blockVal = lane < numWarps ? shared[lane] : -FLT_MAX;
		blockVal = WarpReduceMax(blockVal);
		if(lane == 0){ shared[0] = blockVal; }
	}
	__syncthreads();
	return shared[0];
}
__host__ __device__ inline int NextPow2Clamp(int value){
	if(value <= 0){ return 1; }
	int pow2 = 1;
	while(pow2 < value && pow2 < 256){ pow2 <<= 1; }
	return pow2;
}
__global__ void AttentionPoolScoresKernel(const __half* input, const __half* query, float* scores, int tokens, int embedDim, int numQueries, float invSqrtDim, int totalRows){
	const int stride = blockDim.x * gridDim.x;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalRows * tokens; idx += stride){
		const int row = idx / tokens;
		const int t = idx % tokens;
		const int b = row / numQueries;
		const int q = row % numQueries;
		const __half* inputRow = input + (b * tokens + t) * embedDim;
		const __half* queryRow = query + q * embedDim;
		float sum = 0.0f;
		for(int c = 0; c < embedDim; ++c){ sum = fmaf(__half2float(inputRow[c]), __half2float(queryRow[c]), sum); }
		scores[idx] = sum * invSqrtDim;
	}
}
__global__ void AttentionPoolSoftmaxKernel(float* scores, float* attnWeights, int tokens, int totalRows){
	for(int row = blockIdx.x; row < totalRows; row += gridDim.x){
		float localMax = -FLT_MAX;
		for(int t = threadIdx.x; t < tokens; t += blockDim.x){ localMax = fmaxf(localMax, scores[row * tokens + t]); }
		const float maxVal = BlockReduceMax(localMax);
		__shared__ float sharedMax;
		if(threadIdx.x == 0){ sharedMax = maxVal; }
		__syncthreads();
		const float maxScore = sharedMax;
		float localSum = 0.0f;
		for(int t = threadIdx.x; t < tokens; t += blockDim.x){
			const int idx = row * tokens + t;
			const float expVal = expf(scores[idx] - maxScore);
			attnWeights[idx] = expVal;
			localSum += expVal;
		}
		const float sum = BlockReduceSum(localSum);
		__shared__ float sharedInvSum;
		if(threadIdx.x == 0){ sharedInvSum = sum > 0.0f ? 1.0f / sum : 0.0f; }
		__syncthreads();
		const float invSum = sharedInvSum;
		for(int t = threadIdx.x; t < tokens; t += blockDim.x){ attnWeights[row * tokens + t] *= invSum; }
		__syncthreads();
	}
}
__global__ void AttentionPoolWeightedSumKernel(const __half* input, const float* attnWeights, __half* output, int tokens, int embedDim, int numQueries, int totalRows){
	const int stride = blockDim.x * gridDim.x;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalRows * embedDim; idx += stride){
		const int row = idx / embedDim;
		const int c = idx % embedDim;
		const int b = row / numQueries;
		float sum = 0.0f;
		const float* weightRow = attnWeights + row * tokens;
		const __half* inputCol = input + (b * tokens) * embedDim + c;
		for(int t = 0; t < tokens; ++t){ sum = fmaf(weightRow[t], __half2float(inputCol[t * embedDim]), sum); }
		output[idx] = __float2half(sum);
	}
}
void AttentionPoolForward(const __half* input, const __half* query, __half* output, float* attnWeights, float* tempBuffer, int batchSize, int tokens, int embedDim, int numQueries, float invSqrtDim){
	const int totalRows = batchSize * numQueries;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(static_cast<size_t>(totalRows) * tokens, blocks, tpb);
	AttentionPoolScoresKernel<<<blocks, tpb>>>(input, query, tempBuffer, tokens, embedDim, numQueries, invSqrtDim, totalRows);
	checkCUDA(cudaGetLastError());
	size_t softmaxBlocks = 0, softmaxTpb = 0;
	GetLaunchConfigGridStride(totalRows, softmaxBlocks, softmaxTpb);
	softmaxTpb = static_cast<size_t>(NextPow2Clamp(tokens));
	AttentionPoolSoftmaxKernel<<<softmaxBlocks, softmaxTpb>>>(tempBuffer, attnWeights, tokens, totalRows);
	checkCUDA(cudaGetLastError());
	tpb = 0;
	GetLaunchConfigGridStride(static_cast<size_t>(totalRows) * embedDim, blocks, tpb);
	AttentionPoolWeightedSumKernel<<<blocks, tpb>>>(input, attnWeights, output, tokens, embedDim, numQueries, totalRows);
	checkCUDA(cudaGetLastError());
}
__global__ void AttentionPoolGradWeightsKernel(const __half* grad, const __half* input, float* gradWeights, int tokens, int embedDim, int numQueries, int totalRows){
	const int stride = blockDim.x * gridDim.x;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < totalRows * tokens; idx += stride){
		const int row = idx / tokens;
		const int t = idx % tokens;
		const int b = row / numQueries;
		const int tokenBase = (b * tokens + t) * embedDim;
		const __half* gradVec = grad + row * embedDim;
		const __half* inputVec = input + tokenBase;
		float dot = 0.0f;
		for(int c = 0; c < embedDim; ++c){ dot = fmaf(__half2float(gradVec[c]), __half2float(inputVec[c]), dot); }
		gradWeights[idx] = dot;
	}
}
__global__ void AttentionPoolBatchSumKernel(const float* gradWeights, const float* attnWeights, float* batchSums, int tokens, int totalRows){
	for(int row = blockIdx.x * blockDim.x + threadIdx.x; row < totalRows; row += blockDim.x * gridDim.x){
		float sum = 0.0f;
		for(int t = 0; t < tokens; ++t){ sum += gradWeights[row * tokens + t] * attnWeights[row * tokens + t]; }
		batchSums[row] = sum;
	}
}
__global__ void AttentionPoolGradScoresKernel(float* gradScores, const float* gradWeights, const float* attnWeights, const float* batchSums, int totalRows, int tokens){
	const int total = totalRows * tokens;
	const int stride = blockDim.x * gridDim.x;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int row = idx / tokens;
		gradScores[idx] = attnWeights[idx] * (gradWeights[idx] - batchSums[row]);
	}
}
__global__ void AttentionPoolGradQueryKernel(const float* gradScores, const __half* input, __half* gradQuery, int batchSize, int tokens, int embedDim, int numQueries, float invSqrtDim){
	const int total = numQueries * embedDim;
	const int stride = blockDim.x * gridDim.x;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int q = idx / embedDim;
		const int c = idx % embedDim;
		float sum = 0.0f;
		for(int b = 0; b < batchSize; ++b){
			const float* scoreRow = gradScores + (b * numQueries + q) * tokens;
			const __half* inputPtr = input + (b * tokens) * embedDim + c;
			for(int t = 0; t < tokens; ++t, inputPtr += embedDim){ sum = fmaf(scoreRow[t], __half2float(*inputPtr), sum); }
		}
		gradQuery[idx] = __float2half(sum * invSqrtDim);
	}
}
__global__ void AttentionPoolGradInputKernel(__half* gradInput, const __half* gradOutput, const float* attnWeights, const float* gradScores, const __half* query, int tokens, int embedDim, int numQueries, int totalRows, float invSqrtDim){
	const int total = (totalRows / numQueries) * tokens * embedDim;
	const int stride = blockDim.x * gridDim.x;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int c = idx % embedDim;
		const int bt = idx / embedDim;
		const int t = bt % tokens;
		const int b = bt / tokens;
		float value = 0.0f;
		for(int q = 0; q < numQueries; ++q){
			const int row = b * numQueries + q;
			const float gradY = __half2float(gradOutput[row * embedDim + c]);
			const float weight = attnWeights[row * tokens + t];
			const float scoreGrad = gradScores[row * tokens + t];
			const float scaledQuery = __half2float(query[q * embedDim + c]) * invSqrtDim;
			value = fmaf(scoreGrad, scaledQuery, value + weight * gradY);
		}
		gradInput[idx] = __float2half(value);
	}
}
void AttentionPoolBackward(const __half* grad, const __half* input, const __half* query, const float* attnWeights, float* tempBuffer, float* batchSums, __half* outGrad, __half* gradQuery, int batchSize, int tokens, int embedDim, int numQueries, float invSqrtDim){
	const int totalRows = batchSize * numQueries;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(static_cast<size_t>(totalRows) * tokens, blocks, tpb);
	AttentionPoolGradWeightsKernel<<<blocks, tpb>>>(grad, input, tempBuffer, tokens, embedDim, numQueries, totalRows);
	checkCUDA(cudaGetLastError());
	tpb = 0;
	GetLaunchConfigGridStride(totalRows, blocks, tpb);
	AttentionPoolBatchSumKernel<<<blocks, tpb>>>(tempBuffer, attnWeights, batchSums, tokens, totalRows);
	checkCUDA(cudaGetLastError());
	tpb = 0;
	GetLaunchConfigGridStride(static_cast<size_t>(totalRows) * tokens, blocks, tpb);
	AttentionPoolGradScoresKernel<<<blocks, tpb>>>(tempBuffer, tempBuffer, attnWeights, batchSums, totalRows, tokens);
	checkCUDA(cudaGetLastError());
	tpb = 0;
	GetLaunchConfigGridStride(static_cast<size_t>(numQueries) * embedDim, blocks, tpb);
	AttentionPoolGradQueryKernel<<<blocks, tpb>>>(tempBuffer, input, gradQuery, batchSize, tokens, embedDim, numQueries, invSqrtDim);
	checkCUDA(cudaGetLastError());
	tpb = 0;
	GetLaunchConfigGridStride(static_cast<size_t>(batchSize) * tokens * embedDim, blocks, tpb);
	AttentionPoolGradInputKernel<<<blocks, tpb>>>(outGrad, grad, attnWeights, tempBuffer, query, tokens, embedDim, numQueries, totalRows, invSqrtDim);
	checkCUDA(cudaGetLastError());
}