#include "CuCommon.cuh"
#include <cuda.h>
#include <curand.h>
#include <device_launch_parameters.h>
void BlockShiftHalf(__half* hPtr, const int shiftBy, const int blocksToShift){
	auto blockSize = shiftBy;
	if(blockSize < 0) blockSize = -blockSize;
	if(shiftBy > 0){ for(int i = blocksToShift; 0 < i; --i){ cudaMemcpy(hPtr + i * blockSize, hPtr + (i - 1) * blockSize, blockSize * sizeof(__half), cudaMemcpyDeviceToDevice); } } else{
		for(int i = 0; i < blocksToShift; ++i){ cudaMemcpy(hPtr + (i - 1) * blockSize, hPtr + i * blockSize, blockSize * sizeof(__half), cudaMemcpyDeviceToDevice); }
	}
}
__global__ void GradientKernel(__half* grads, const __half* predictions, const __half* targets, const float clip, const int size){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){ grads[idx] = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx] - targets[idx])))); }
}
void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const float clip, const int size){
	auto gridSize = DivCeil(size, BS);
	GradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, size);
}
__global__ void SplitGradKernel(__half* gradients, const __half* predictions, const float* targets, const float clip, const int numCtrls, const int numButs, const int batchSize, const int size){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx / numCtrls;
		const int ctrlId = idx % numCtrls;
		const auto diff = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx]) - targets[idx])));
		if(ctrlId < numButs){
			const auto gradIdx = batchId * numButs + ctrlId;
			gradients[gradIdx] = diff;
		} else{
			const auto gradIdx = numButs * batchSize + batchId * (numCtrls - numButs) + (ctrlId - numButs);
			gradients[gradIdx] = diff;
		}
	}
}
void SplitGradient(__half* dGradient, const __half* dPredictions, const float* dTargets, const float clip, const int size, const int numCtrls, const int numButs, const int batchSize){
	auto gridSize = DivCeil(size, BS);
	SplitGradKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, numCtrls, numButs, batchSize, size);
}
__global__ void MergeOutputsKernel(__half* predOut, const __half* buttonData, const __half* axisData, const int size, const int numCtrls, const int numButs){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx / numCtrls;
		const int ctrlId = idx % numCtrls;
		if(ctrlId < numButs){ predOut[idx] = buttonData[batchId * numButs + ctrlId]; } else{ predOut[idx] = axisData[batchId * (numCtrls - numButs) + (ctrlId - numButs)]; }
	}
}
void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, const int numCtrls, const int numButs, const int size){
	auto gridSize = DivCeil(size, BS);
	MergeOutputsKernel<<<gridSize, BS>>>(predOut, buttonData, axisData, size, numCtrls, numButs);
}
__global__ void BCEGradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size, const float scale){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size){
		const float y = __half2float(targets[idx]);
		const float pClamped = fminf(fmaxf(__half2float(predictions[idx]), EPSILON_F), 1.0f - EPSILON_F);
		float gradient = 0.0f;
		if(y == 1.0f){
			gradient = (pClamped - 1.0f) / pClamped;
			gradients[idx] = __float2half(gradient * scale);
		} else{
			gradient = pClamped / (1.0f - pClamped);
			gradients[idx] = __float2half(gradient * scale);
		}
	}
}
void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size, const float scale){
	auto gridSize = DivCeil(size, BS);
	BCEGradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, scale);
}
__device__ int deviceResult;
__global__ void isNaNKernel(const __half* __restrict__ data, int size){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < size && __hisnan(data[idx])){ atomicExch(&deviceResult, 1); }
}
bool IsnanHalf(const __half* __restrict__ data, int size){
	int hResult = 0;
	cudaMemcpyToSymbol(deviceResult, &hResult, sizeof(int));
	auto gridSize = DivCeil(size, BS);
	isNaNKernel<<<gridSize, BS>>>(data, size);
	cudaMemcpyFromSymbol(&hResult, deviceResult, sizeof(int));
	return hResult != 0;
}
__global__ void FeatureMapMosaicKernel(const __half* __restrict__ input, unsigned char* __restrict__ output, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, const float scale){
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.z * blockDim.z + threadIdx.z;
	if(c >= inC || y >= H || x >= W) return;
	const int tileX = c % gridW;
	const int tileY = c / gridW;
	const int outX = tileX * tileW + x;
	const int outY = tileY * tileH + y;
	const __half value = input[c * H * W + y * W + x];
	const float fVal = __half2float(value);
	const unsigned char pixel = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, fVal * 255.0f * scale)));
	output[outY * mosaicW + outX] = pixel;
}
void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, const float scale, cudaStream_t stream){
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((inC + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, (W + blockDim.z - 1) / blockDim.z);
	FeatureMapMosaicKernel<<<gridDim, blockDim, 0, stream>>>(dInput, dOutput, H, W, inC, mosaicW, tileW, tileH, gridW, scale);
}
__global__ void GetPredictionKernel(const __half* predBatch, float* prediction, const int numCtrls, const int size){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < numCtrls){ prediction[idx] = __half2float(predBatch[idx + size - numCtrls]); }
}
void GetPrediction(const __half* predBatch, float* prediction, const int numCtrls, const int batchSize){
	float* devPtr = nullptr;
	cudaHostGetDevicePointer(&devPtr, prediction, 0);
	GetPredictionKernel<<<1, numCtrls>>>(predBatch, devPtr, numCtrls, batchSize * numCtrls);
	cudaDeviceSynchronize();
}
__global__ void AttentionPoolScoresKernel(const __half* input, const __half* query, float* scores, int batchSize, int tokens, int embedDim, float invSqrtDim){
	const int b = blockIdx.x;
	const int t = blockIdx.y * blockDim.x + threadIdx.x;
	if(b >= batchSize || t >= tokens){ return; }
	const int base = (b * tokens + t) * embedDim;
	float sum = 0.0f;
	for(int c = 0; c < embedDim; ++c){ sum += __half2float(input[base + c]) * __half2float(query[c]); }
	scores[b * tokens + t] = sum * invSqrtDim;
}
__global__ void AttentionPoolSoftmaxKernel(float* scores, float* attnWeights, int batchSize, int tokens){
	const int b = blockIdx.x;
	if(b >= batchSize){ return; }
	float maxVal = scores[b * tokens];
	for(int t = 1; t < tokens; ++t){ maxVal = fmaxf(maxVal, scores[b * tokens + t]); }
	float sum = 0.0f;
	for(int t = 0; t < tokens; ++t){
		const float expVal = expf(scores[b * tokens + t] - maxVal);
		attnWeights[b * tokens + t] = expVal;
		sum += expVal;
	}
	const float invSum = sum > 0.0f ? 1.0f / sum : 0.0f;
	for(int t = 0; t < tokens; ++t){ attnWeights[b * tokens + t] *= invSum; }
}
__global__ void AttentionPoolWeightedSumKernel(const __half* input, const float* attnWeights, __half* output, int batchSize, int tokens, int embedDim){
	const int b = blockIdx.x;
	const int c = blockIdx.y * blockDim.x + threadIdx.x;
	if(b >= batchSize || c >= embedDim){ return; }
	float sum = 0.0f;
	for(int t = 0; t < tokens; ++t){
		const float weight = attnWeights[b * tokens + t];
		const int idx = (b * tokens + t) * embedDim + c;
		sum += weight * __half2float(input[idx]);
	}
	output[b * embedDim + c] = __float2half(sum);
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
	const int t = blockIdx.y * blockDim.x + threadIdx.x;
	if(b >= batchSize || t >= tokens){ return; }
	const int tokenBase = (b * tokens + t) * embedDim;
	const int gradBase = b * embedDim;
	float dot = 0.0f;
	for(int c = 0; c < embedDim; ++c){ dot += __half2float(grad[gradBase + c]) * __half2float(input[tokenBase + c]); }
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
	for(int b = 0; b < batchSize; ++b){
		for(int t = 0; t < tokens; ++t){
			const int idx = (b * tokens + t) * embedDim + c;
			sum += gradScores[b * tokens + t] * __half2float(input[idx]);
		}
	}
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
	const float queryVal = __half2float(query[c]);
	const float value = weight * gradY + scoreGrad * queryVal * invSqrtDim;
	gradInput[idx] = __float2half(value);
}
void AttentionPoolBackward(const __half* grad, const __half* input, const __half* query, const float* attnWeights, float* tempBuffer, float* batchSums, __half* outGrad, __half* gradQuery, int batchSize, int tokens, int embedDim, float invSqrtDim){
	dim3 gradWeightGrid(batchSize, DivCeil(tokens, BS));
	AttentionPoolGradWeightsKernel<<<gradWeightGrid, BS>>>(grad, input, tempBuffer, batchSize, tokens, embedDim);
	AttentionPoolBatchSumKernel<<<batchSize, 1>>>(tempBuffer, attnWeights, batchSums, batchSize, tokens);
	const int totalTokens = batchSize * tokens;
	AttentionPoolGradScoresKernel<<<DivCeil(totalTokens, BS), BS>>>(tempBuffer, tempBuffer, attnWeights, batchSums, batchSize, tokens);
	AttentionPoolGradQueryKernel<<<DivCeil(embedDim, BS), BS>>>(tempBuffer, input, gradQuery, batchSize, tokens, embedDim, invSqrtDim);
	dim3 gradInputGrid(batchSize, tokens, DivCeil(embedDim, BS));
	AttentionPoolGradInputKernel<<<gradInputGrid, BS>>>(outGrad, grad, attnWeights, tempBuffer, query, batchSize, tokens, embedDim, invSqrtDim);
}
__global__ void ScaleHalfKernel(__half* data, const size_t count, const float scale){
	const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count; idx += stride){ data[idx] = __float2half(__half2float(data[idx]) * scale); }
}
void ScaleArrayHalf(__half* data, const size_t count, const float scale){
	if(!data || scale == 1.0f || count == 0) return;
	constexpr int bs = 256;
	const int blocks = DivCeil(count, bs);
	ScaleHalfKernel<<<blocks, bs>>>(data, count, scale);
	checkCUDA(cudaGetLastError());
}
__global__ void AddBiasKernel(__half* output, const __half* bias, const int channels, const int batch){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = channels * batch;
	if(idx >= total){ return; }
	const int c = idx % channels;
	output[idx] = output[idx] + bias[c];
}
void AddBias(__half* output, const __half* bias, const int channels, const int batch){
	constexpr int bs = 256;
	const int blocks = DivCeil(channels * batch, bs);
	AddBiasKernel<<<blocks, bs>>>(output, bias, channels, batch);
	checkCUDA(cudaGetLastError());
}
__global__ void AccumulateBiasGradKernel(const __half* grad, __half* gradBias, const int channels, const int batch, const float scale, const bool reset){
	const int c = blockIdx.x * blockDim.x + threadIdx.x;
	if(c >= channels){ return; }
	float sum = 0.0f;
	for(int b = 0; b < batch; ++b){ sum += __half2float(grad[c + b * channels]); }
	const float scaled = sum * scale;
	if(reset){ gradBias[c] = __float2half(scaled); } else{ gradBias[c] = __float2half(__half2float(gradBias[c]) + scaled); }
}
void AccumulateBiasGrad(const __half* grad, __half* gradBias, const int channels, const int batch, const float scale, const bool reset){
	constexpr int bs = 256;
	const int blocks = DivCeil(channels, bs);
	AccumulateBiasGradKernel<<<blocks, bs>>>(grad, gradBias, channels, batch, scale, reset);
	checkCUDA(cudaGetLastError());
}
__global__ void TokensToSpatialKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch) * tokens * embedDim;
	if(idx >= total) return;
	const int feature = idx % embedDim;
	const int tokenIndex = (idx / embedDim) % tokens;
	const int batchIndex = idx / (embedDim * tokens);
	const int row = tokenIndex / patchCols;
	const int col = tokenIndex % patchCols;
	const size_t outIdx = (((static_cast<size_t>(batchIndex) * embedDim + feature) * patchRows) + row) * patchCols + col;
	output[outIdx] = input[idx];
}
void TokensToSpatial(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t total = static_cast<size_t>(batch) * tokens * embedDim;
	int blocks = 0;
	int threads = 0;
	GetLaunchConfig(static_cast<int>(total), blocks, threads);
	TokensToSpatialKernel<<<blocks, threads>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}
__global__ void SpatialToTokensKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch) * embedDim * patchRows * patchCols;
	if(idx >= total) return;
	const int col = idx % patchCols;
	const int row = (idx / patchCols) % patchRows;
	const int feature = (idx / (patchCols * patchRows)) % embedDim;
	const int batchIndex = idx / (static_cast<size_t>(embedDim) * patchRows * patchCols);
	const int tokenIndex = row * patchCols + col;
	const size_t outIdx = ((static_cast<size_t>(batchIndex) * tokens + tokenIndex) * embedDim) + feature;
	output[outIdx] = input[idx];
}
void SpatialToTokens(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t total = static_cast<size_t>(batch) * embedDim * patchRows * patchCols;
	int blocks = 0;
	int threads = 0;
	GetLaunchConfig(static_cast<int>(total), blocks, threads);
	SpatialToTokensKernel<<<blocks, threads>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}