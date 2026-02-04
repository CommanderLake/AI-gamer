#include "CuCommon.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
__global__ void DropPathBuildMaskKernel(float* mask, int batch, float keepProb, float scale){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= batch) return;
	const float value = mask[idx];
	mask[idx] = value < keepProb ? scale : 0.0f;
}
void DropPathBuildMask(float* mask, int batch, float keepProb){
	if(batch <= 0) return;
	constexpr int bs = 256;
	const auto blocks = DivCeil(batch, bs);
	const float scale = keepProb > 0.0f ? 1.0f/keepProb : 0.0f;
	DropPathBuildMaskKernel<<<blocks, bs>>>(mask, batch, keepProb, scale);
	checkCUDA(cudaGetLastError());
}
__global__ void DropPathApplyKernel(__half* data, const float* mask, int batch, int elementsPerBatch){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch) * elementsPerBatch;
	if(idx >= total) return;
	const int batchIndex = idx / elementsPerBatch;
	const float scale = mask[batchIndex];
	data[idx] = __float2half(__half2float(data[idx]) * scale);
}
void DropPathApply(__half* data, const float* mask, int batch, int elementsPerBatch){
	if(batch <= 0 || elementsPerBatch <= 0) return;
	const size_t total = static_cast<size_t>(batch) * elementsPerBatch;
	constexpr int bs = 256;
	const auto blocks = DivCeil(static_cast<int>(total), bs);
	DropPathApplyKernel<<<blocks, bs>>>(data, mask, batch, elementsPerBatch);
	checkCUDA(cudaGetLastError());
}