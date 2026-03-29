#define __CUDACC__
#include "CuCommon.cuh"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
__device__ __forceinline__ unsigned int DropoutHash(unsigned int x){
	x ^= x >> 17;
	x *= 0xed5ad4bbU;
	x ^= x >> 11;
	x *= 0xac4c1b51U;
	x ^= x >> 15;
	x *= 0x31848babU;
	x ^= x >> 14;
	return x;
}
__device__ __forceinline__ float DropoutRandomUniform(const unsigned long long seed, const int idx){
	const unsigned int low = static_cast<unsigned int>(seed);
	const unsigned int high = static_cast<unsigned int>(seed >> 32);
	const unsigned int mixed = DropoutHash(static_cast<unsigned int>(idx) ^ low) ^ DropoutHash(high + 0x9e3779b9U + static_cast<unsigned int>(idx));
	return static_cast<float>(mixed) * (1.0f / 4294967296.0f);
}
__global__ void DropPathBuildMaskKernel(float* mask, const int batch, const float keepProb, const float scale){
	const auto stride = blockDim.x * gridDim.x;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < batch; idx += stride){
		const float value = mask[idx];
		mask[idx] = value < keepProb ? scale : 0.0f;
	}
}
void DropPathBuildMask(float* mask, const int batch, const float keepProb){
	if(batch <= 0) return;
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(batch, blocks, tpb);
	const float scale = keepProb > 0.0f ? 1.0f / keepProb : 0.0f;
	DropPathBuildMaskKernel<<<blocks, tpb>>>(mask, batch, keepProb, scale);
	checkCUDA(cudaGetLastError());
}
__global__ void DropPathApplyKernel(__half* data, const float* mask, const int batch, const int elementsPerBatch){
	const size_t total = static_cast<size_t>(batch) * elementsPerBatch;
	const auto stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	for(size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int batchIndex = idx / elementsPerBatch;
		const float scale = mask[batchIndex];
		data[idx] = __float2half(__half2float(data[idx]) * scale);
	}
}
void DropPathApply(__half* data, const float* mask, const int batch, const int elementsPerBatch){
	if(batch <= 0 || elementsPerBatch <= 0) return;
	const size_t total = static_cast<size_t>(batch) * elementsPerBatch;
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(total, blocks, tpb);
	DropPathApplyKernel<<<blocks, tpb>>>(data, mask, batch, elementsPerBatch);
	checkCUDA(cudaGetLastError());
}
__global__ void DropoutForwardKernel(__half* data, unsigned char* mask, const int size, const float keepProb, const float scale, const unsigned long long seed){
	const int pairCount = (size + 1) >> 1;
	const auto stride = blockDim.x * gridDim.x;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < pairCount; idx += stride){
		const int idx2 = idx << 1;
		const float r0 = DropoutRandomUniform(seed, idx2);
		const unsigned char m0 = r0 < keepProb ? 1 : 0;
		mask[idx2] = m0;
		if(idx2 + 1 < size){
			const float r1 = DropoutRandomUniform(seed, idx2 + 1);
			const unsigned char m1 = r1 < keepProb ? 1 : 0;
			mask[idx2 + 1] = m1;
			const __half2 in = *reinterpret_cast<const __half2*>(data + idx2);
			const __half2 scaleVec = __floats2half2_rn(m0 ? scale : 0.0f, m1 ? scale : 0.0f);
			*reinterpret_cast<__half2*>(data + idx2) = __hmul2(in, scaleVec);
		} else{
			float v0 = __half2float(data[idx2]);
			v0 = m0 ? v0 * scale : 0.0f;
			data[idx2] = __float2half(v0);
		}
	}
}
void DropoutForward(__half* data, unsigned char* mask, const int size, const float keepProb, const unsigned long long seed){
	if(size <= 0 || keepProb <= 0.0f) return;
	const int elements2 = DivCeil(size, 2);
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(elements2, blocks, tpb);
	const float scale = 1.0f / keepProb;
	DropoutForwardKernel<<<blocks, tpb>>>(data, mask, size, keepProb, scale, seed);
	checkCUDA(cudaGetLastError());
}
__global__ void DropoutBackwardKernel(__half* grad, const unsigned char* mask, const int size, const float scale){
	const int pairCount = (size + 1) >> 1;
	const auto stride = blockDim.x * gridDim.x;
	for(int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < pairCount; idx += stride){
		const int idx2 = idx << 1;
		if(idx2 + 1 < size){
			const __half2 in = *reinterpret_cast<const __half2*>(grad + idx2);
			const __half2 scaleVec = __floats2half2_rn(mask[idx2] ? scale : 0.0f, mask[idx2 + 1] ? scale : 0.0f);
			*reinterpret_cast<__half2*>(grad + idx2) = __hmul2(in, scaleVec);
		} else{
			float g0 = __half2float(grad[idx2]);
			grad[idx2] = __float2half(mask[idx2] ? g0 * scale : 0.0f);
		}
	}
}
void DropoutBackward(__half* grad, const unsigned char* mask, const int size, const float keepProb){
	if(size <= 0 || keepProb <= 0.0f) return;
	const int elements2 = DivCeil(size, 2);
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(elements2, blocks, tpb);
	const float scale = 1.0f / keepProb;
	DropoutBackwardKernel<<<blocks, tpb>>>(grad, mask, size, scale);
	checkCUDA(cudaGetLastError());
}