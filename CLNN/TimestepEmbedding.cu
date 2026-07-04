#define __CUDACC__
#include "CuCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
__device__ __forceinline__ float TimestepEmbeddingFreq(const int dim, const int halfDim, const float maxPeriod){
	if(halfDim <= 0) return 1.0f;
	return expf(-logf(maxPeriod)*static_cast<float>(dim)/static_cast<float>(halfDim));
}
__global__ void TimestepEmbeddingFloatKernel(__half* __restrict__ out, const float* __restrict__ timesteps, int batchSize, int embeddingDim, float maxPeriod){
	const int total = batchSize*embeddingDim;
	const int halfDim = embeddingDim/2;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += blockDim.x*gridDim.x){
		const int b = idx/embeddingDim;
		const int d = idx%embeddingDim;
		float value = 0.0f;
		if(d < halfDim){
			value = cosf(timesteps[b]*TimestepEmbeddingFreq(d, halfDim, maxPeriod));
		} else if(d < 2*halfDim){
			value = sinf(timesteps[b]*TimestepEmbeddingFreq(d - halfDim, halfDim, maxPeriod));
		}
		out[idx] = __float2half(value);
	}
}
__global__ void TimestepEmbeddingHalfKernel(__half* __restrict__ out, const __half* __restrict__ timesteps, int batchSize, int embeddingDim, float maxPeriod){
	const int total = batchSize*embeddingDim;
	const int halfDim = embeddingDim/2;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += blockDim.x*gridDim.x){
		const int b = idx/embeddingDim;
		const int d = idx%embeddingDim;
		const float timestep = __half2float(timesteps[b]);
		float value = 0.0f;
		if(d < halfDim){
			value = cosf(timestep*TimestepEmbeddingFreq(d, halfDim, maxPeriod));
		} else if(d < 2*halfDim){
			value = sinf(timestep*TimestepEmbeddingFreq(d - halfDim, halfDim, maxPeriod));
		}
		out[idx] = __float2half(value);
	}
}
void TimestepEmbeddingForward(__half* out, const float* timesteps, int batchSize, int embeddingDim, float maxPeriod){
	if(!out || !timesteps){
		fprintf(stderr, "TimestepEmbeddingForward: Null pointer input\n");
		return;
	}
	if(batchSize <= 0 || embeddingDim <= 0 || maxPeriod <= 1.0f){
		fprintf(stderr, "TimestepEmbeddingForward: Invalid arguments batch=%d, dim=%d, maxPeriod=%f\n", batchSize, embeddingDim, maxPeriod);
		return;
	}
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(static_cast<size_t>(batchSize)*embeddingDim, blocks, tpb);
	TimestepEmbeddingFloatKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(out, timesteps, batchSize, embeddingDim, maxPeriod);
	checkCUDA(cudaGetLastError());
}
void TimestepEmbeddingForwardHalf(__half* out, const __half* timesteps, int batchSize, int embeddingDim, float maxPeriod){
	if(!out || !timesteps){
		fprintf(stderr, "TimestepEmbeddingForwardHalf: Null pointer input\n");
		return;
	}
	if(batchSize <= 0 || embeddingDim <= 0 || maxPeriod <= 1.0f){
		fprintf(stderr, "TimestepEmbeddingForwardHalf: Invalid arguments batch=%d, dim=%d, maxPeriod=%f\n", batchSize, embeddingDim, maxPeriod);
		return;
	}
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(static_cast<size_t>(batchSize)*embeddingDim, blocks, tpb);
	TimestepEmbeddingHalfKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(out, timesteps, batchSize, embeddingDim, maxPeriod);
	checkCUDA(cudaGetLastError());
}
