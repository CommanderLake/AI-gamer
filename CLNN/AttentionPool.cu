#define __CUDACC__
#include "CuCommon.cuh"
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
__global__ void AttentionPoolScoresKernel(const __half* input, const __half* query, float* scores, int batchSize, int tokens, int embedDim, float invSqrtDim){
	const int b = blockIdx.x;
	const int t = blockIdx.y * blockDim.x + threadIdx.x;
	if(b >= batchSize || t >= tokens){ return; }
	const int base = (b * tokens + t) * embedDim;
	const __half* inputRow = input + base;
	float sum = 0.0f;
	for(int c = 0; c < embedDim; ++c){ sum = fmaf(__half2float(inputRow[c]), __half2float(query[c]), sum); }
	scores[b * tokens + t] = sum * invSqrtDim;
}
__global__ void AttentionPoolSoftmaxKernel(float* scores, float* attnWeights, int batchSize, int tokens){
	const int b = blockIdx.x;
	if(b >= batchSize){ return; }
	float localMax = -FLT_MAX;
	for(int t = threadIdx.x; t < tokens; t += blockDim.x){ localMax = fmaxf(localMax, scores[b * tokens + t]); }
	const float maxVal = BlockReduceMax(localMax);
	__shared__ float sharedMax;
	if(threadIdx.x == 0){ sharedMax = maxVal; }
	__syncthreads();
	const float maxScore = sharedMax;
	float localSum = 0.0f;
	for(int t = threadIdx.x; t < tokens; t += blockDim.x){
		const int idx = b * tokens + t;
		const float expVal = expf(scores[idx] - maxScore);
		attnWeights[idx] = expVal;
		localSum += expVal;
	}
	const float sum = BlockReduceSum(localSum);
	__shared__ float sharedInvSum;
	if(threadIdx.x == 0){ sharedInvSum = sum > 0.0f ? 1.0f / sum : 0.0f; }
	__syncthreads();
	const float invSum = sharedInvSum;
	for(int t = threadIdx.x; t < tokens; t += blockDim.x){ attnWeights[b * tokens + t] *= invSum; }
}
__global__ void AttentionPoolWeightedSumKernel(const __half* input, const float* attnWeights, __half* output, int batchSize, int tokens, int embedDim){
	const int b = blockIdx.x;
	const int c = blockIdx.y * blockDim.x + threadIdx.x;
	if(b >= batchSize || c >= embedDim){ return; }
	float sum = 0.0f;
	const float* weightRow = attnWeights + b * tokens;
	const __half* inputCol = input + (b * tokens) * embedDim + c;
	for(int t = 0; t < tokens; ++t){ sum = fmaf(weightRow[t], __half2float(inputCol[t * embedDim]), sum); }
	output[b * embedDim + c] = __float2half(sum);
}
void AttentionPoolForward(const __half* input, const __half* query, __half* output, float* attnWeights, float* tempBuffer, int batchSize, int tokens, int embedDim, float invSqrtDim){
	dim3 scoreGrid(batchSize, DivCeil(tokens, CPM));
	AttentionPoolScoresKernel<<<scoreGrid, CPM>>>(input, query, tempBuffer, batchSize, tokens, embedDim, invSqrtDim);
	checkCUDA(cudaGetLastError());
	int softmaxThreads = NextPow2Clamp(tokens);
	AttentionPoolSoftmaxKernel<<<batchSize, softmaxThreads>>>(tempBuffer, attnWeights, batchSize, tokens);
	checkCUDA(cudaGetLastError());
	dim3 sumGrid(batchSize, DivCeil(embedDim, CPM));
	AttentionPoolWeightedSumKernel<<<sumGrid, CPM>>>(input, attnWeights, output, batchSize, tokens, embedDim);
	checkCUDA(cudaGetLastError());
}
__global__ void AttentionPoolGradWeightsKernel(const __half* grad, const __half* input, float* gradWeights, int batchSize, int tokens, int embedDim){
	const int b = blockIdx.x;
	const int t = blockIdx.y * blockDim.x + threadIdx.x;
	if(b >= batchSize || t >= tokens){ return; }
	const int tokenBase = (b * tokens + t) * embedDim;
	const __half* gradVec = grad + b * embedDim;
	const __half* inputVec = input + tokenBase;
	float dot = 0.0f;
	for(int c = 0; c < embedDim; ++c){ dot = fmaf(__half2float(gradVec[c]), __half2float(inputVec[c]), dot); }
	gradWeights[b * tokens + t] = dot;
}
__global__ void AttentionPoolBatchSumKernel(const float* gradWeights, const float* attnWeights, float* batchSums, int batchSize, int tokens){
	const int b = blockIdx.x;
	if(b >= batchSize){ return; }
	float sum = 0.0f;
	for(int t = 0; t < tokens; ++t){ sum += gradWeights[b * tokens + t] * attnWeights[b * tokens + t]; }
	batchSums[b] = sum;
}
__global__ void AttentionPoolGradScoresKernel(float* gradScores, const float* gradWeights, const float* attnWeights, const float* batchSums, int batchSize, int tokens){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = batchSize * tokens;
	if(idx >= total){ return; }
	const int b = idx / tokens;
	gradScores[idx] = attnWeights[idx] * (gradWeights[idx] - batchSums[b]);
}
__global__ void AttentionPoolGradQueryKernel(const float* gradScores, const __half* input, __half* gradQuery, int batchSize, int tokens, int embedDim, float invSqrtDim){
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if(c >= embedDim){ return; }
	float sum = 0.0f;
	const int totalTokens = batchSize * tokens;
	const __half* inputPtr = input + c;
	for(int idx = 0; idx < totalTokens; ++idx, inputPtr += embedDim){ sum = fmaf(gradScores[idx], __half2float(*inputPtr), sum); }
	gradQuery[c] = __float2half(sum * invSqrtDim);
}
__global__ void AttentionPoolGradInputKernel(__half* gradInput, const __half* gradOutput, const float* attnWeights, const float* gradScores, const __half* query, int batchSize, int tokens, int embedDim, float invSqrtDim){
	const int b = blockIdx.x;
	const int t = blockIdx.y;
	const int c = blockIdx.z * blockDim.x + threadIdx.x;
	if(b >= batchSize || t >= tokens || c >= embedDim){ return; }
	const int idx = (b * tokens + t) * embedDim + c;
	const float gradY = __half2float(gradOutput[b * embedDim + c]);
	const float weight = attnWeights[b * tokens + t];
	const float scoreGrad = gradScores[b * tokens + t];
	const float scaledQuery = __half2float(query[c]) * invSqrtDim;
	const float value = fmaf(scoreGrad, scaledQuery, weight * gradY);
	gradInput[idx] = __float2half(value);
}
void AttentionPoolBackward(const __half* grad, const __half* input, const __half* query, const float* attnWeights, float* tempBuffer, float* batchSums, __half* outGrad, __half* gradQuery, int batchSize, int tokens, int embedDim, float invSqrtDim){
	dim3 gradWeightGrid(batchSize, DivCeil(tokens, CPM));
	AttentionPoolGradWeightsKernel<<<gradWeightGrid, CPM>>>(grad, input, tempBuffer, batchSize, tokens, embedDim);
	checkCUDA(cudaGetLastError());
	AttentionPoolBatchSumKernel<<<batchSize, 1>>>(tempBuffer, attnWeights, batchSums, batchSize, tokens);
	checkCUDA(cudaGetLastError());
	const int totalTokens = batchSize * tokens;
	AttentionPoolGradScoresKernel<<<DivCeil(totalTokens, CPM), CPM>>>(tempBuffer, tempBuffer, attnWeights, batchSums, batchSize, tokens);
	checkCUDA(cudaGetLastError());
	AttentionPoolGradQueryKernel<<<DivCeil(embedDim, CPM), CPM>>>(tempBuffer, input, gradQuery, batchSize, tokens, embedDim, invSqrtDim);
	checkCUDA(cudaGetLastError());
	dim3 gradInputGrid(batchSize, tokens, DivCeil(embedDim, CPM));
	AttentionPoolGradInputKernel<<<gradInputGrid, CPM>>>(outGrad, grad, attnWeights, tempBuffer, query, batchSize, tokens, embedDim, invSqrtDim);
	checkCUDA(cudaGetLastError());
}