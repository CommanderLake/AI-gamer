#include "CuCommon.cuh"
#include <cuda.h>
#include <curand.h>
#include <device_launch_parameters.h>
__global__ void AttentionPoolScoresKernel(const __half* input, const __half* query, float* scores, int batchSize, int tokens, int embedDim, float invSqrtDim){
	const int b = blockIdx.x;
	const int t = blockIdx.y*blockDim.x + threadIdx.x;
	if(b >= batchSize || t >= tokens){ return; }
	const int base = (b*tokens + t)*embedDim;
	float sum = 0.0f;
	for(int c = 0; c < embedDim; ++c){ sum += __half2float(input[base + c])*__half2float(query[c]); }
	scores[b*tokens + t] = sum*invSqrtDim;
}
__global__ void AttentionPoolSoftmaxKernel(float* scores, float* attnWeights, int batchSize, int tokens){
	const int b = blockIdx.x;
	if(b >= batchSize){ return; }
	float maxVal = scores[b*tokens];
	for(int t = 1; t < tokens; ++t){ maxVal = fmaxf(maxVal, scores[b*tokens + t]); }
	float sum = 0.0f;
	for(int t = 0; t < tokens; ++t){
		const float expVal = expf(scores[b*tokens + t] - maxVal);
		attnWeights[b*tokens + t] = expVal;
		sum += expVal;
	}
	const float invSum = sum > 0.0f ? 1.0f / sum : 0.0f;
	for(int t = 0; t < tokens; ++t){ attnWeights[b*tokens + t] *= invSum; }
}
__global__ void AttentionPoolWeightedSumKernel(const __half* input, const float* attnWeights, __half* output, int batchSize, int tokens, int embedDim){
	const int b = blockIdx.x;
	const int c = blockIdx.y*blockDim.x + threadIdx.x;
	if(b >= batchSize || c >= embedDim){ return; }
	float sum = 0.0f;
	for(int t = 0; t < tokens; ++t){
		const float weight = attnWeights[b*tokens + t];
		const int idx = (b*tokens + t)*embedDim + c;
		sum += weight*__half2float(input[idx]);
	}
	output[b*embedDim + c] = __float2half(sum);
}
void AttentionPoolForward(const __half* input, const __half* query, __half* output, float* attnWeights, float* tempBuffer, int batchSize, int tokens, int embedDim, float invSqrtDim){
	dim3 scoreGrid(batchSize, DivCeil(tokens, BS));
	AttentionPoolScoresKernel<<<scoreGrid, BS>>>(input, query, tempBuffer, batchSize, tokens, embedDim, invSqrtDim);
	AttentionPoolSoftmaxKernel<<<batchSize, 1>>>(tempBuffer, attnWeights, batchSize, tokens);
	dim3 sumGrid(batchSize, DivCeil(embedDim, BS));
	AttentionPoolWeightedSumKernel<<<sumGrid, BS>>>(input, attnWeights, output, batchSize, tokens, embedDim);
}
__global__ void AttentionPoolGradWeightsKernel(const __half* grad, const __half* input, float* gradWeights, int batchSize, int tokens, int embedDim){
	const int b = blockIdx.x;
	const int t = blockIdx.y*blockDim.x + threadIdx.x;
	if(b >= batchSize || t >= tokens){ return; }
	const int tokenBase = (b*tokens + t)*embedDim;
	const int gradBase = b*embedDim;
	float dot = 0.0f;
	for(int c = 0; c < embedDim; ++c){ dot += __half2float(grad[gradBase + c])*__half2float(input[tokenBase + c]); }
	gradWeights[b*tokens + t] = dot;
}
__global__ void AttentionPoolBatchSumKernel(const float* gradWeights, const float* attnWeights, float* batchSums, int batchSize, int tokens){
	const int b = blockIdx.x;
	if(b >= batchSize){ return; }
	float sum = 0.0f;
	for(int t = 0; t < tokens; ++t){ sum += gradWeights[b*tokens + t]*attnWeights[b*tokens + t]; }
	batchSums[b] = sum;
}
__global__ void AttentionPoolGradScoresKernel(float* gradScores, const float* gradWeights, const float* attnWeights, const float* batchSums, int batchSize, int tokens){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = batchSize*tokens;
	if(idx >= total){ return; }
	const int b = idx / tokens;
	gradScores[idx] = attnWeights[idx]*(gradWeights[idx] - batchSums[b]);
}
__global__ void AttentionPoolGradQueryKernel(const float* gradScores, const __half* input, __half* gradQuery, int batchSize, int tokens, int embedDim, float invSqrtDim){
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	if(c >= embedDim){ return; }
	float sum = 0.0f;
	for(int b = 0; b < batchSize; ++b){
		for(int t = 0; t < tokens; ++t){
			const int idx = (b*tokens + t)*embedDim + c;
			sum += gradScores[b*tokens + t]*__half2float(input[idx]);
		}
	}
	gradQuery[c] = __float2half(sum*invSqrtDim);
}
__global__ void AttentionPoolGradInputKernel(__half* gradInput, const __half* gradOutput, const float* attnWeights, const float* gradScores, const __half* query, int batchSize, int tokens, int embedDim, float invSqrtDim){
	const int b = blockIdx.x;
	const int t = blockIdx.y;
	const int c = blockIdx.z*blockDim.x + threadIdx.x;
	if(b >= batchSize || t >= tokens || c >= embedDim){ return; }
	const int idx = (b*tokens + t)*embedDim + c;
	const float gradY = __half2float(gradOutput[b*embedDim + c]);
	const float weight = attnWeights[b*tokens + t];
	const float scoreGrad = gradScores[b*tokens + t];
	const float queryVal = __half2float(query[c]);
	const float value = weight*gradY + scoreGrad*queryVal*invSqrtDim;
	gradInput[idx] = __float2half(value);
}
void AttentionPoolBackward(const __half* grad, const __half* input, const __half* query, const float* attnWeights, float* tempBuffer, float* batchSums, __half* outGrad, __half* gradQuery, int batchSize, int tokens, int embedDim, float invSqrtDim){
	dim3 gradWeightGrid(batchSize, DivCeil(tokens, BS));
	AttentionPoolGradWeightsKernel<<<gradWeightGrid, BS>>>(grad, input, tempBuffer, batchSize, tokens, embedDim);
	AttentionPoolBatchSumKernel<<<batchSize, 1>>>(tempBuffer, attnWeights, batchSums, batchSize, tokens);
	const int totalTokens = batchSize*tokens;
	AttentionPoolGradScoresKernel<<<DivCeil(totalTokens, BS), BS>>>(tempBuffer, tempBuffer, attnWeights, batchSums, batchSize, tokens);
	AttentionPoolGradQueryKernel<<<DivCeil(embedDim, BS), BS>>>(tempBuffer, input, gradQuery, batchSize, tokens, embedDim, invSqrtDim);
	dim3 gradInputGrid(batchSize, tokens, DivCeil(embedDim, BS));
	AttentionPoolGradInputKernel<<<gradInputGrid, BS>>>(outGrad, grad, attnWeights, tempBuffer, query, batchSize, tokens, embedDim, invSqrtDim);
}