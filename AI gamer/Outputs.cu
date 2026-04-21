#include "CuCommon.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
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