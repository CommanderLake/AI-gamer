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
__global__ void TimestepEmbeddingBackwardKernel(float* __restrict__ gradTimesteps, const __half* __restrict__ grad, const float* __restrict__ timesteps, int batchSize, int embeddingDim, float maxPeriod){
	const int b = blockIdx.x;
	if(b >= batchSize) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<float*>(smem);
	const int halfDim = embeddingDim/2;
	const float timestep = timesteps[b];
	float sum = 0.0f;
	for(int d = tid; d < embeddingDim; d += blockDim.x){
		float derivative = 0.0f;
		if(d < halfDim){
			const float freq = TimestepEmbeddingFreq(d, halfDim, maxPeriod);
			derivative = -sinf(timestep*freq)*freq;
		} else if(d < 2*halfDim){
			const float freq = TimestepEmbeddingFreq(d - halfDim, halfDim, maxPeriod);
			derivative = cosf(timestep*freq)*freq;
		}
		sum = fmaf(__half2float(grad[b*embeddingDim + d]), derivative, sum);
	}
#pragma unroll
	for(int offset = 16; offset > 0; offset >>= 1){ sum += __shfl_down_sync(0xFFFFFFFF, sum, offset); }
	if(laneId == 0){ warpBuffer[warpId] = sum; }
	__syncthreads();
	if(warpId == 0){
		float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
#pragma unroll
		for(int offset = 16; offset > 0; offset >>= 1){ blockSum += __shfl_down_sync(0xFFFFFFFF, blockSum, offset); }
		if(laneId == 0){ gradTimesteps[b] = blockSum; }
	}
}
__global__ void TimestepEmbeddingBackwardHalfKernel(__half* __restrict__ gradTimesteps, const __half* __restrict__ grad, const __half* __restrict__ timesteps, int batchSize, int embeddingDim, float maxPeriod){
	const int b = blockIdx.x;
	if(b >= batchSize) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<float*>(smem);
	const int halfDim = embeddingDim/2;
	const float timestep = __half2float(timesteps[b]);
	float sum = 0.0f;
	for(int d = tid; d < embeddingDim; d += blockDim.x){
		float derivative = 0.0f;
		if(d < halfDim){
			const float freq = TimestepEmbeddingFreq(d, halfDim, maxPeriod);
			derivative = -sinf(timestep*freq)*freq;
		} else if(d < 2*halfDim){
			const float freq = TimestepEmbeddingFreq(d - halfDim, halfDim, maxPeriod);
			derivative = cosf(timestep*freq)*freq;
		}
		sum = fmaf(__half2float(grad[b*embeddingDim + d]), derivative, sum);
	}
#pragma unroll
	for(int offset = 16; offset > 0; offset >>= 1){ sum += __shfl_down_sync(0xFFFFFFFF, sum, offset); }
	if(laneId == 0){ warpBuffer[warpId] = sum; }
	__syncthreads();
	if(warpId == 0){
		float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
#pragma unroll
		for(int offset = 16; offset > 0; offset >>= 1){ blockSum += __shfl_down_sync(0xFFFFFFFF, blockSum, offset); }
		if(laneId == 0){ gradTimesteps[b] = __float2half(blockSum); }
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
void TimestepEmbeddingBackward(float* gradTimesteps, const __half* grad, const float* timesteps, int batchSize, int embeddingDim, float maxPeriod){
	if(!gradTimesteps || !grad || !timesteps){
		fprintf(stderr, "TimestepEmbeddingBackward: Null pointer input\n");
		return;
	}
	if(batchSize <= 0 || embeddingDim <= 0 || maxPeriod <= 1.0f){
		fprintf(stderr, "TimestepEmbeddingBackward: Invalid arguments batch=%d, dim=%d, maxPeriod=%f\n", batchSize, embeddingDim, maxPeriod);
		return;
	}
	int threads = 32;
	while(threads < embeddingDim && threads < 256){ threads <<= 1; }
	const int warpsPerBlock = (threads + 31)/32;
	const size_t smemSize = warpsPerBlock*sizeof(float);
	TimestepEmbeddingBackwardKernel<<<batchSize, threads, smemSize>>>(gradTimesteps, grad, timesteps, batchSize, embeddingDim, maxPeriod);
	checkCUDA(cudaGetLastError());
}
void TimestepEmbeddingBackwardHalf(__half* gradTimesteps, const __half* grad, const __half* timesteps, int batchSize, int embeddingDim, float maxPeriod){
	if(!gradTimesteps || !grad || !timesteps){
		fprintf(stderr, "TimestepEmbeddingBackwardHalf: Null pointer input\n");
		return;
	}
	if(batchSize <= 0 || embeddingDim <= 0 || maxPeriod <= 1.0f){
		fprintf(stderr, "TimestepEmbeddingBackwardHalf: Invalid arguments batch=%d, dim=%d, maxPeriod=%f\n", batchSize, embeddingDim, maxPeriod);
		return;
	}
	int threads = 32;
	while(threads < embeddingDim && threads < 256){ threads <<= 1; }
	const int warpsPerBlock = (threads + 31)/32;
	const size_t smemSize = warpsPerBlock*sizeof(float);
	TimestepEmbeddingBackwardHalfKernel<<<batchSize, threads, smemSize>>>(gradTimesteps, grad, timesteps, batchSize, embeddingDim, maxPeriod);
	checkCUDA(cudaGetLastError());
}
