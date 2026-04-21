#include "APICommon.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda_fp16.h>
__global__ void MergeOutputsKernel(__half* predOut, const __half* buttonData, const __half* axisData, const int size, const int numCtrls, const int numButs){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx < size){
		const int batchId = idx/numCtrls;
		const int ctrlId = idx % numCtrls;
		if(ctrlId < numButs){ predOut[idx] = buttonData[batchId*numButs + ctrlId]; } else{ predOut[idx] = axisData[batchId*(numCtrls - numButs) + (ctrlId - numButs)]; }
	}
}
void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, const int numCtrls, const int numButs, const int size){
	auto gridSize = DivCeil(size, 256);
	MergeOutputsKernel<<<gridSize, 256>>>(predOut, buttonData, axisData, size, numCtrls, numButs);
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
__global__ void SelectLastTemporalFrameKernel(const __half* input, __half* output, int batchSize, int temporalLength, int featureSize){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = batchSize*featureSize;
	if(idx >= total) return;
	const int b = idx/featureSize;
	const int f = idx - b*featureSize;
	const int srcIndex = ((b*temporalLength + (temporalLength - 1))*featureSize) + f;
	output[idx] = input[srcIndex];
}
void SelectLastTemporalFrame(const __half* input, __half* output, const int batchSize, const int temporalLength, const int featureSize){
	if(batchSize <= 0 || temporalLength <= 0 || featureSize <= 0) return;
	const int total = batchSize*featureSize;
	SelectLastTemporalFrameKernel<<<DivCeil(total, 256), 256>>>(input, output, batchSize, temporalLength, featureSize);
	checkCUDA(cudaGetLastError());
}
__global__ void ExpandTemporalOutputsKernel(const __half* input, __half* output, int batchSize, int temporalLength, int featureSize){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = batchSize*temporalLength*featureSize;
	if(idx >= total) return;
	const int f = idx % featureSize;
	const int b = idx/(featureSize*temporalLength);
	output[idx] = input[b*featureSize + f];
}
void ExpandTemporalOutputs(const __half* input, __half* output, const int batchSize, const int temporalLength, const int featureSize){
	if(batchSize <= 0 || temporalLength <= 0 || featureSize <= 0) return;
	const int total = batchSize*temporalLength*featureSize;
	ExpandTemporalOutputsKernel<<<DivCeil(total, 256), 256>>>(input, output, batchSize, temporalLength, featureSize);
	checkCUDA(cudaGetLastError());
}
__global__ void ReduceTemporalGradientsKernel(const __half* inputGrad, __half* reducedGrad, int batchSize, int temporalLength, int featureSize){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = batchSize*featureSize;
	if(idx >= total) return;
	const int b = idx/featureSize;
	const int f = idx - b*featureSize;
	float sum = 0.0f;
	for(int t = 0; t < temporalLength; ++t){
		sum += __half2float(inputGrad[(b*temporalLength + t)*featureSize + f]);
	}
	reducedGrad[idx] = __float2half(sum);
}
void ReduceTemporalGradients(const __half* inputGrad, __half* reducedGrad, const int batchSize, const int temporalLength, const int featureSize){
	if(batchSize <= 0 || temporalLength <= 0 || featureSize <= 0) return;
	const int total = batchSize*featureSize;
	ReduceTemporalGradientsKernel<<<DivCeil(total, 256), 256>>>(inputGrad, reducedGrad, batchSize, temporalLength, featureSize);
	checkCUDA(cudaGetLastError());
}
__global__ void ScatterLastTemporalFrameGradKernel(const __half* input, __half* output, int batchSize, int temporalLength, int featureSize){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = batchSize*temporalLength*featureSize;
	if(idx >= total) return;
	const int f = idx % featureSize;
	const int t = (idx/featureSize) % temporalLength;
	const int b = idx/(featureSize*temporalLength);
	output[idx] = (t == temporalLength - 1) ? input[b*featureSize + f] : __float2half(0.0f);
}
void ScatterLastTemporalFrameGrad(const __half* input, __half* output, const int batchSize, const int temporalLength, const int featureSize){
	if(batchSize <= 0 || temporalLength <= 0 || featureSize <= 0) return;
	const int total = batchSize*temporalLength*featureSize;
	ScatterLastTemporalFrameGradKernel<<<DivCeil(total, 256), 256>>>(input, output, batchSize, temporalLength, featureSize);
	checkCUDA(cudaGetLastError());
}
