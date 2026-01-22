#define __CUDACC__
#include "CuCommon.cuh"
#include <cuda.h>
#include <curand.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
void BlockShiftHalf(__half* hPtr, const int shiftBy, const int blocksToShift){
	if(!hPtr || shiftBy == 0 || blocksToShift <= 1){
		return;
	}
	auto blockSize = shiftBy;
	if(blockSize < 0) blockSize = -blockSize;
	if(shiftBy > 0){
		for(int i = blocksToShift - 1; i > 0; --i){
			checkCUDA(cudaMemcpy(hPtr + i*blockSize, hPtr + (i - 1)*blockSize, blockSize*sizeof(__half), cudaMemcpyDeviceToDevice));
		}
	} else{
		for(int i = 0; i + 1 < blocksToShift; ++i){
			checkCUDA(cudaMemcpy(hPtr + (i - 1)*blockSize, hPtr + i*blockSize, blockSize*sizeof(__half), cudaMemcpyDeviceToDevice));
		}
	}
}
__global__ void ShiftTokens2dKernel(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int shiftY, const int shiftX){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = batch*height*width*channels;
	if(idx >= total) return;
	const int c = idx % channels;
	const int token = idx / channels;
	const int x = token % width;
	const int y = (token / width) % height;
	const int b = token / (width*height);
	const int srcY = (y + shiftY + height) % height;
	const int srcX = (x + shiftX + width) % width;
	const int srcToken = (b*height + srcY)*width + srcX;
	output[idx] = input[srcToken*channels + c];
}
void ShiftTokens2d(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int shiftY, const int shiftX){
	const int total = batch*height*width*channels;
	const int gridSize = DivCeil(total, BS);
	ShiftTokens2dKernel<<<gridSize, BS>>>(input, output, batch, height, width, channels, shiftY, shiftX);
	checkCUDA(cudaGetLastError());
}
__global__ void WindowPartitionKernel(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int windowsPerRow){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int windowTokens = windowSize*windowSize;
	const int total = batch*windowsPerRow*(height/windowSize)*windowTokens*channels;
	if(idx >= total) return;
	const int c = idx % channels;
	const int token = (idx / channels) % windowTokens;
	const int windowIndex = (idx / channels) / windowTokens % (windowsPerRow*(height/windowSize));
	const int b = idx / (channels*windowTokens*windowsPerRow*(height/windowSize));
	const int localX = token % windowSize;
	const int localY = token / windowSize;
	const int windowX = windowIndex % windowsPerRow;
	const int windowY = windowIndex / windowsPerRow;
	const int srcX = windowX*windowSize + localX;
	const int srcY = windowY*windowSize + localY;
	const int srcToken = (b*height + srcY)*width + srcX;
	output[idx] = input[srcToken*channels + c];
}
__global__ void WindowPartitionKernelHalf2(const __half2* input, __half2* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int windowsPerRow){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int half2Channels = channels / 2;
	const int windowTokens = windowSize*windowSize;
	const int total = batch*windowsPerRow*(height/windowSize)*windowTokens*half2Channels;
	if(idx >= total) return;
	const int c2 = idx % half2Channels;
	const int token = (idx / half2Channels) % windowTokens;
	const int windowIndex = (idx / half2Channels) / windowTokens % (windowsPerRow*(height/windowSize));
	const int b = idx / (half2Channels*windowTokens*windowsPerRow*(height/windowSize));
	const int localX = token % windowSize;
	const int localY = token / windowSize;
	const int windowX = windowIndex % windowsPerRow;
	const int windowY = windowIndex / windowsPerRow;
	const int srcX = windowX*windowSize + localX;
	const int srcY = windowY*windowSize + localY;
	const int srcToken = (b*height + srcY)*width + srcX;
	output[idx] = input[srcToken*half2Channels + c2];
}
void WindowPartition(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int windowSize){
	const int windowsPerRow = width / windowSize;
	const int windowsPerCol = height / windowSize;
	const int total = batch*windowsPerRow*windowsPerCol*windowSize*windowSize*channels;
	const int gridSize = DivCeil(total, BS);
	const bool useHalf2 = (channels % 2 == 0) && ((reinterpret_cast<uintptr_t>(input) % alignof(__half2)) == 0) && ((reinterpret_cast<uintptr_t>(output) % alignof(__half2)) == 0);
	if(useHalf2){
		const int totalHalf2 = total / 2;
		const int gridHalf2 = DivCeil(totalHalf2, BS);
		WindowPartitionKernelHalf2<<<gridHalf2, BS>>>(reinterpret_cast<const __half2*>(input), reinterpret_cast<__half2*>(output), batch, height, width, channels, windowSize, windowsPerRow);
	} else{
		WindowPartitionKernel<<<gridSize, BS>>>(input, output, batch, height, width, channels, windowSize, windowsPerRow);
	}
	checkCUDA(cudaGetLastError());
}
__global__ void WindowReverseKernel(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int windowsPerRow){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int windowTokens = windowSize*windowSize;
	const int total = batch*windowsPerRow*(height/windowSize)*windowTokens*channels;
	if(idx >= total) return;
	const int c = idx % channels;
	const int token = (idx / channels) % windowTokens;
	const int windowIndex = (idx / channels) / windowTokens % (windowsPerRow*(height/windowSize));
	const int b = idx / (channels*windowTokens*windowsPerRow*(height/windowSize));
	const int localX = token % windowSize;
	const int localY = token / windowSize;
	const int windowX = windowIndex % windowsPerRow;
	const int windowY = windowIndex / windowsPerRow;
	const int dstX = windowX*windowSize + localX;
	const int dstY = windowY*windowSize + localY;
	const int dstToken = (b*height + dstY)*width + dstX;
	output[dstToken*channels + c] = input[idx];
}
__global__ void WindowReverseKernelHalf2(const __half2* input, __half2* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int windowsPerRow){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int half2Channels = channels / 2;
	const int windowTokens = windowSize*windowSize;
	const int total = batch*windowsPerRow*(height/windowSize)*windowTokens*half2Channels;
	if(idx >= total) return;
	const int c2 = idx % half2Channels;
	const int token = (idx / half2Channels) % windowTokens;
	const int windowIndex = (idx / half2Channels) / windowTokens % (windowsPerRow*(height/windowSize));
	const int b = idx / (half2Channels*windowTokens*windowsPerRow*(height/windowSize));
	const int localX = token % windowSize;
	const int localY = token / windowSize;
	const int windowX = windowIndex % windowsPerRow;
	const int windowY = windowIndex / windowsPerRow;
	const int dstX = windowX*windowSize + localX;
	const int dstY = windowY*windowSize + localY;
	const int dstToken = (b*height + dstY)*width + dstX;
	output[dstToken*half2Channels + c2] = input[idx];
}
void WindowReverse(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int windowSize){
	const int windowsPerRow = width / windowSize;
	const int windowsPerCol = height / windowSize;
	const int total = batch*windowsPerRow*windowsPerCol*windowSize*windowSize*channels;
	const int gridSize = DivCeil(total, BS);
	const bool useHalf2 = (channels % 2 == 0) && ((reinterpret_cast<uintptr_t>(input) % alignof(__half2)) == 0) && ((reinterpret_cast<uintptr_t>(output) % alignof(__half2)) == 0);
	if(useHalf2){
		const int totalHalf2 = total / 2;
		const int gridHalf2 = DivCeil(totalHalf2, BS);
		WindowReverseKernelHalf2<<<gridHalf2, BS>>>(reinterpret_cast<const __half2*>(input), reinterpret_cast<__half2*>(output), batch, height, width, channels, windowSize, windowsPerRow);
	} else{
		WindowReverseKernel<<<gridSize, BS>>>(input, output, batch, height, width, channels, windowSize, windowsPerRow);
	}
	checkCUDA(cudaGetLastError());
}
__global__ void ShiftWindowPartitionKernel(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int shiftY, const int shiftX, const int windowsPerRow){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int windowTokens = windowSize*windowSize;
	const int total = batch*windowsPerRow*(height/windowSize)*windowTokens*channels;
	if(idx >= total) return;
	const int c = idx % channels;
	const int token = (idx / channels) % windowTokens;
	const int windowIndex = (idx / channels) / windowTokens % (windowsPerRow*(height/windowSize));
	const int b = idx / (channels*windowTokens*windowsPerRow*(height/windowSize));
	const int localX = token % windowSize;
	const int localY = token / windowSize;
	const int windowX = windowIndex % windowsPerRow;
	const int windowY = windowIndex / windowsPerRow;
	const int dstX = windowX*windowSize + localX;
	const int dstY = windowY*windowSize + localY;
	const int srcY = (dstY + shiftY + height) % height;
	const int srcX = (dstX + shiftX + width) % width;
	const int srcToken = (b*height + srcY)*width + srcX;
	output[idx] = input[srcToken*channels + c];
}
__global__ void ShiftWindowPartitionKernelHalf2(const __half2* input, __half2* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int shiftY, const int shiftX, const int windowsPerRow){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int half2Channels = channels / 2;
	const int windowTokens = windowSize*windowSize;
	const int total = batch*windowsPerRow*(height/windowSize)*windowTokens*half2Channels;
	if(idx >= total) return;
	const int c2 = idx % half2Channels;
	const int token = (idx / half2Channels) % windowTokens;
	const int windowIndex = (idx / half2Channels) / windowTokens % (windowsPerRow*(height/windowSize));
	const int b = idx / (half2Channels*windowTokens*windowsPerRow*(height/windowSize));
	const int localX = token % windowSize;
	const int localY = token / windowSize;
	const int windowX = windowIndex % windowsPerRow;
	const int windowY = windowIndex / windowsPerRow;
	const int dstX = windowX*windowSize + localX;
	const int dstY = windowY*windowSize + localY;
	const int srcY = (dstY + shiftY + height) % height;
	const int srcX = (dstX + shiftX + width) % width;
	const int srcToken = (b*height + srcY)*width + srcX;
	output[idx] = input[srcToken*half2Channels + c2];
}
void ShiftWindowPartition(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int shiftY, const int shiftX){
	const int windowsPerRow = width / windowSize;
	const int windowsPerCol = height / windowSize;
	const int total = batch*windowsPerRow*windowsPerCol*windowSize*windowSize*channels;
	const int gridSize = DivCeil(total, BS);
	const bool useHalf2 = (channels % 2 == 0) && ((reinterpret_cast<uintptr_t>(input) % alignof(__half2)) == 0) && ((reinterpret_cast<uintptr_t>(output) % alignof(__half2)) == 0);
	if(useHalf2){
		const int totalHalf2 = total / 2;
		const int gridHalf2 = DivCeil(totalHalf2, BS);
		ShiftWindowPartitionKernelHalf2<<<gridHalf2, BS>>>(reinterpret_cast<const __half2*>(input), reinterpret_cast<__half2*>(output), batch, height, width, channels, windowSize, shiftY, shiftX, windowsPerRow);
	} else{
		ShiftWindowPartitionKernel<<<gridSize, BS>>>(input, output, batch, height, width, channels, windowSize, shiftY, shiftX, windowsPerRow);
	}
	checkCUDA(cudaGetLastError());
}
__global__ void WindowReverseShiftKernel(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int shiftY, const int shiftX, const int windowsPerRow){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int windowTokens = windowSize*windowSize;
	const int total = batch*windowsPerRow*(height/windowSize)*windowTokens*channels;
	if(idx >= total) return;
	const int c = idx % channels;
	const int token = (idx / channels) % windowTokens;
	const int windowIndex = (idx / channels) / windowTokens % (windowsPerRow*(height/windowSize));
	const int b = idx / (channels*windowTokens*windowsPerRow*(height/windowSize));
	const int localX = token % windowSize;
	const int localY = token / windowSize;
	const int windowX = windowIndex % windowsPerRow;
	const int windowY = windowIndex / windowsPerRow;
	const int srcX = windowX*windowSize + localX;
	const int srcY = windowY*windowSize + localY;
	const int dstY = (srcY + shiftY + height) % height;
	const int dstX = (srcX + shiftX + width) % width;
	const int dstToken = (b*height + dstY)*width + dstX;
	output[dstToken*channels + c] = input[idx];
}
__global__ void WindowReverseShiftKernelHalf2(const __half2* input, __half2* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int shiftY, const int shiftX, const int windowsPerRow){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int half2Channels = channels / 2;
	const int windowTokens = windowSize*windowSize;
	const int total = batch*windowsPerRow*(height/windowSize)*windowTokens*half2Channels;
	if(idx >= total) return;
	const int c2 = idx % half2Channels;
	const int token = (idx / half2Channels) % windowTokens;
	const int windowIndex = (idx / half2Channels) / windowTokens % (windowsPerRow*(height/windowSize));
	const int b = idx / (half2Channels*windowTokens*windowsPerRow*(height/windowSize));
	const int localX = token % windowSize;
	const int localY = token / windowSize;
	const int windowX = windowIndex % windowsPerRow;
	const int windowY = windowIndex / windowsPerRow;
	const int srcX = windowX*windowSize + localX;
	const int srcY = windowY*windowSize + localY;
	const int dstY = (srcY + shiftY + height) % height;
	const int dstX = (srcX + shiftX + width) % width;
	const int dstToken = (b*height + dstY)*width + dstX;
	output[dstToken*half2Channels + c2] = input[idx];
}
void WindowReverseShift(const __half* input, __half* output, const int batch, const int height, const int width, const int channels, const int windowSize, const int shiftY, const int shiftX){
	const int windowsPerRow = width / windowSize;
	const int windowsPerCol = height / windowSize;
	const int total = batch*windowsPerRow*windowsPerCol*windowSize*windowSize*channels;
	const int gridSize = DivCeil(total, BS);
	const bool useHalf2 = (channels % 2 == 0) && ((reinterpret_cast<uintptr_t>(input) % alignof(__half2)) == 0) && ((reinterpret_cast<uintptr_t>(output) % alignof(__half2)) == 0);
	if(useHalf2){
		const int totalHalf2 = total / 2;
		const int gridHalf2 = DivCeil(totalHalf2, BS);
		WindowReverseShiftKernelHalf2<<<gridHalf2, BS>>>(reinterpret_cast<const __half2*>(input), reinterpret_cast<__half2*>(output), batch, height, width, channels, windowSize, shiftY, shiftX, windowsPerRow);
	} else{
		WindowReverseShiftKernel<<<gridSize, BS>>>(input, output, batch, height, width, channels, windowSize, shiftY, shiftX, windowsPerRow);
	}
	checkCUDA(cudaGetLastError());
}
__global__ void PackTokens2x2Kernel(const __half* input, __half* output, const int batch, const int height, const int width, const int channels){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int outHeight = height / 2;
	const int outWidth = width / 2;
	const int outTokens = outHeight * outWidth;
	const int total = batch*outTokens*channels*4;
	if(idx >= total) return;
	const int c = idx % (channels * 4);
	const int outToken = (idx / (channels * 4)) % outTokens;
	const int b = idx / (channels * 4 * outTokens);
	const int outX = outToken % outWidth;
	const int outY = outToken / outWidth;
	const int quad = c / channels;
	const int cIn = c % channels;
	const int inX = outX * 2 + (quad % 2);
	const int inY = outY * 2 + (quad / 2);
	const int inToken = (b * height + inY) * width + inX;
	output[idx] = input[inToken * channels + cIn];
}
void PackTokens2x2(const __half* input, __half* output, const int batch, const int height, const int width, const int channels){
	const int outHeight = height / 2;
	const int outWidth = width / 2;
	const int total = batch*outHeight*outWidth*channels*4;
	const int gridSize = DivCeil(total, BS);
	PackTokens2x2Kernel<<<gridSize, BS>>>(input, output, batch, height, width, channels);
	checkCUDA(cudaGetLastError());
}
__global__ void UnpackTokens2x2Kernel(const __half* input, __half* output, const int batch, const int height, const int width, const int channels){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int inTokens = height * width;
	const int total = batch*inTokens*channels*4;
	if(idx >= total) return;
	const int c = idx % (channels * 4);
	const int inToken = (idx / (channels * 4)) % inTokens;
	const int b = idx / (channels * 4 * inTokens);
	const int inX = inToken % width;
	const int inY = inToken / width;
	const int quad = c / channels;
	const int cIn = c % channels;
	const int outX = inX * 2 + (quad % 2);
	const int outY = inY * 2 + (quad / 2);
	const int outToken = (b * (height * 2) + outY) * (width * 2) + outX;
	output[outToken * channels + cIn] = input[idx];
}
void UnpackTokens2x2(const __half* input, __half* output, const int batch, const int height, const int width, const int channels){
	const int total = batch*height*width*channels*4;
	const int gridSize = DivCeil(total, BS);
	UnpackTokens2x2Kernel<<<gridSize, BS>>>(input, output, batch, height, width, channels);
	checkCUDA(cudaGetLastError());
}
__global__ void AccumulateRelPosBiasGradKernel(const float* dAtt, const int* relPosIndex, float* gradBias, const int batch, const int heads, const int tokens, const int biasSize, const float scale){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = batch*heads*tokens*tokens;
	if(idx >= total) return;
	const int token = idx % (tokens*tokens);
	const int head = (idx / (tokens*tokens)) % heads;
	const int biasIdx = relPosIndex[token];
	if(biasIdx >= 0 && biasIdx < biasSize){
		atomicAdd(&gradBias[head*biasSize + biasIdx], dAtt[idx]*scale);
	}
}
void AccumulateRelPosBiasGrad(const float* dAtt, const int* relPosIndex, float* gradBias, const int batch, const int heads, const int tokens, const int biasSize, const float scale){
	if(!dAtt || !relPosIndex || !gradBias) return;
	const int total = batch*heads*tokens*tokens;
	const int gridSize = DivCeil(total, BS);
	AccumulateRelPosBiasGradKernel<<<gridSize, BS>>>(dAtt, relPosIndex, gradBias, batch, heads, tokens, biasSize, scale);
	checkCUDA(cudaGetLastError());
}
__global__ void GradientKernel(__half* grads, const __half* predictions, const __half* targets, const float clip, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){ grads[idx] = __float2half(fmaxf(-clip, fminf(clip, __half2float(predictions[idx] - targets[idx])))); }
}
void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const float clip, const int size){
	auto gridSize = DivCeil(size, BS);
	GradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, clip, size);
	checkCUDA(cudaGetLastError());
}
__device__ inline float Sigmoidf(const float x){
	if(x >= 0.0f){
		const float z = __expf(-x);
		return 1.0f / (1.0f + z);
	}
	const float z = __expf(x);
	return z / (1.0f + z);
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
			const int axisId = ctrlId - numButs;
			const int numAxisOutputs = numCtrls - numButs;
			const int numAxes = numAxisOutputs / 2;
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
__global__ void MergeOutputsKernel(__half* predOut, const __half* buttonData, const __half* axisData, const int size, const int numCtrls, const int numButs){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx / numCtrls;
		const int ctrlId = idx % numCtrls;
		if(ctrlId < numButs){ predOut[idx] = buttonData[batchId*numButs + ctrlId]; } else{ predOut[idx] = axisData[batchId*(numCtrls - numButs) + (ctrlId - numButs)]; }
	}
}
void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, const int numCtrls, const int numButs, const int size){
	auto gridSize = DivCeil(size, BS);
	MergeOutputsKernel<<<gridSize, BS>>>(predOut, buttonData, axisData, size, numCtrls, numButs);
	checkCUDA(cudaGetLastError());
}
__global__ void BCEGradientKernel(__half* gradients, const __half* predictions, const __half* targets, const int size, const float scale){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size) gradients[idx] = __float2half((Sigmoidf(__half2float(predictions[idx])) - __half2float(targets[idx]))*scale);
}
void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, const int size, const float scale){
	auto gridSize = DivCeil(size, BS);
	BCEGradientKernel<<<gridSize, BS>>>(dGradient, dPredictions, dTargets, size, scale);
	checkCUDA(cudaGetLastError());
}
__device__ int deviceResult;
__global__ void isNaNKernel(const __half* __restrict__ data, int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size && __hisnan(data[idx])){ atomicExch(&deviceResult, 1); }
}
bool IsnanHalf(const __half* __restrict__ data, int size){
	int hResult = 0;
	cudaMemcpyToSymbol(deviceResult, &hResult, sizeof(int));
	auto gridSize = DivCeil(size, BS);
	isNaNKernel<<<gridSize, BS>>>(data, size);
	checkCUDA(cudaGetLastError());
	cudaMemcpyFromSymbol(&hResult, deviceResult, sizeof(int));
	return hResult != 0;
}
__global__ void FeatureMapMosaicKernel(const __half* __restrict__ input, unsigned char* __restrict__ output, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, const float scale){
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	const int y = blockIdx.y*blockDim.y + threadIdx.y;
	const int x = blockIdx.z*blockDim.z + threadIdx.z;
	if(c >= inC || y >= H || x >= W) return;
	const int tileX = c % gridW;
	const int tileY = c / gridW;
	const int outX = tileX*tileW + x;
	const int outY = tileY*tileH + y;
	const float fVal = __half2float(input[c*H*W + y*W + x]);
	const unsigned char pixel = static_cast<unsigned char>(fmaxf(0.0f, fminf(255.0f, fVal*255.0f*scale)));
	output[outY*mosaicW + outX] = pixel;
}
void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, const int H, const int W, const int inC, const int mosaicW, const int tileW, const int tileH, const int gridW, const float scale, cudaStream_t stream){
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((inC + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, (W + blockDim.z - 1) / blockDim.z);
	FeatureMapMosaicKernel<<<gridDim, blockDim, 0, stream>>>(dInput, dOutput, H, W, inC, mosaicW, tileW, tileH, gridW, scale);
	checkCUDA(cudaGetLastError());
}
__global__ void GetPredictionKernel(const __half* predBatch, float* prediction, const int numCtrls, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < numCtrls){ prediction[idx] = __half2float(predBatch[idx + size - numCtrls]); }
}
void GetPrediction(const __half* predBatch, float* prediction, const int numCtrls, const int batchSize){
	float* devPtr = nullptr;
	cudaHostGetDevicePointer(&devPtr, prediction, 0);
	GetPredictionKernel<<<1, numCtrls>>>(predBatch, devPtr, numCtrls, batchSize*numCtrls);
	checkCUDA(cudaGetLastError());
	cudaDeviceSynchronize();
}
__global__ void ScaleHalfKernel(__half* data, const size_t count, const float scale){
	const size_t stride = static_cast<size_t>(blockDim.x)*gridDim.x;
	for(size_t idx = blockIdx.x*blockDim.x + threadIdx.x; idx < count; idx += stride){ data[idx] = __float2half(__half2float(data[idx])*scale); }
}
void ScaleArrayHalf(__half* data, const size_t count, const float scale){
	if(!data || scale == 1.0f || count == 0) return;
	constexpr int bs = 256;
	const auto blocks = DivCeil(count, bs);
	ScaleHalfKernel<<<blocks, bs>>>(data, count, scale);
	checkCUDA(cudaGetLastError());
}
__global__ void AddBiasKernel(__half* output, const __half* bias, const int channels, const int batchSize){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = channels*batchSize;
	if(idx >= total){ return; }
	const int c = idx % channels;
	output[idx] = output[idx] + bias[c];
}
void AddBias(__half* output, const __half* bias, const int channels, const int batchSize){
	constexpr int bs = 256;
	const auto blocks = DivCeil(channels*batchSize, bs);
	AddBiasKernel<<<blocks, bs>>>(output, bias, channels, batchSize);
	checkCUDA(cudaGetLastError());
}
__global__ void AddTensorKernel(__half alpha, __half* A, __half beta, const __half* B, const int size){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= size){ return; }
	A[idx] = __hfma(alpha, A[idx], __hfma(beta, B[idx], 0));
}
void AddTensor(float alpha, __half* A, float beta, const __half* B, const int size){
	constexpr int bs = 256;
	const auto blocks = DivCeil(size, bs);
	AddTensorKernel<<<blocks, bs>>>(__half(alpha), A, __half(beta), B, size);
	checkCUDA(cudaGetLastError());
}
__global__ void AccumulateBiasGradKernel(const __half* grad, __half* gradBias, const int channels, const int batch, const float scale, const bool reset){
	const int c = blockIdx.x*blockDim.x + threadIdx.x;
	if(c >= channels){ return; }
	float sum = 0.0f;
	for(int b = 0; b < batch; ++b){ sum += __half2float(grad[c + b*channels]); }
	const float scaled = sum*scale;
	if(reset){ gradBias[c] = __float2half(scaled); } else{ gradBias[c] = __float2half(__half2float(gradBias[c]) + scaled); }
}
void AccumulateBiasGrad(const __half* grad, __half* gradBias, const int channels, const int batch, const float scale, const bool reset){
	constexpr int bs = 256;
	const auto blocks = DivCeil(channels, bs);
	AccumulateBiasGradKernel<<<blocks, bs>>>(grad, gradBias, channels, batch, scale, reset);
	checkCUDA(cudaGetLastError());
}
__global__ void TokensToSpatialKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	if(idx >= total) return;
	const int feature = idx % embedDim;
	const int tokenIndex = idx / embedDim % tokens;
	const int batchIndex = idx / (static_cast<size_t>(embedDim)*tokens);
	const int row = tokenIndex / patchCols;
	const int col = tokenIndex % patchCols;
	const size_t outIdx = ((static_cast<size_t>(batchIndex)*embedDim + feature)*patchRows + row)*patchCols + col;
	output[outIdx] = input[idx];
}
void TokensToSpatial(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	int bs = 256;
	const auto blocks = DivCeil(total, bs);
	TokensToSpatialKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}
__global__ void SpatialToTokensKernel(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*embedDim*patchRows*patchCols;
	if(idx >= total) return;
	const int col = idx % patchCols;
	const int row = idx / patchCols % patchRows;
	const int feature = idx / (patchCols*patchRows) % embedDim;
	const int batchIndex = idx / (static_cast<size_t>(embedDim)*patchRows*patchCols);
	const int tokenIndex = row*patchCols + col;
	const size_t outIdx = (static_cast<size_t>(batchIndex)*tokens + tokenIndex)*embedDim + feature;
	output[outIdx] = input[idx];
}
void SpatialToTokens(const __half* input, __half* output, int batch, int tokens, int embedDim, int patchRows, int patchCols){
	const size_t total = static_cast<size_t>(batch)*embedDim*patchRows*patchCols;
	int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	SpatialToTokensKernel<<<blocks, bs>>>(input, output, batch, tokens, embedDim, patchRows, patchCols);
	checkCUDA(cudaGetLastError());
}
