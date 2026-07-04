#define __CUDACC__
#include "CuCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#define RMS_GRAD_CLIP 5.0f
__device__ __forceinline__ float RMSWarpReduceSum(float value){
#pragma unroll
	for(int offset = 16; offset > 0; offset >>= 1){ value += __shfl_down_sync(0xFFFFFFFF, value, offset); }
	return value;
}
int SelectRMSNormThreads(const int elements){
	int threads = 32;
	while(threads < elements && threads < 512) threads <<= 1;
	if(elements < 32){ threads = 32; }
	return threads;
}
__global__ void RMSNormForwardKernel(__half* __restrict__ y, const __half* __restrict__ x, const float* __restrict__ gamma, float* __restrict__ invRms, int N, int C, int HW, float epsilon){
	const int n = blockIdx.x;
	if(n >= N) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<float*>(smem);
	__shared__ float sharedInvRms;
	const int stride = C*HW;
	const int base = n*stride;
	float sumSq = 0.0f;
	for(int i = tid; i < stride; i += blockDim.x){
		const float v = __half2float(x[base + i]);
		sumSq = fmaf(v, v, sumSq);
	}
	sumSq = RMSWarpReduceSum(sumSq);
	if(laneId == 0){ warpBuffer[warpId] = sumSq; }
	__syncthreads();
	if(warpId == 0){
		float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
		blockSum = RMSWarpReduceSum(blockSum);
		if(laneId == 0){
			sharedInvRms = rsqrtf(blockSum/fmaxf(static_cast<float>(stride), 1.0f) + epsilon);
			invRms[n] = sharedInvRms;
		}
	}
	__syncthreads();
	for(int i = tid; i < stride; i += blockDim.x){
		const int c = i/HW;
		const float v = __half2float(x[base + i]);
		y[base + i] = __float2half(v*sharedInvRms*__ldg(gamma + c));
	}
}
__global__ void RMSNormForwardSpatialKernel(__half* __restrict__ y, const __half* __restrict__ x, const float* __restrict__ gamma, float* __restrict__ invRms, int N, int C, int HW, float epsilon){
	const int n = blockIdx.y;
	if(n >= N) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<float*>(smem);
	__shared__ float sharedInvRms;
	for(int hw = blockIdx.x; hw < HW; hw += gridDim.x){
		float sumSq = 0.0f;
		const int base = n*C*HW + hw;
		for(int c = tid; c < C; c += blockDim.x){
			const float v = __half2float(x[base + c*HW]);
			sumSq = fmaf(v, v, sumSq);
		}
		sumSq = RMSWarpReduceSum(sumSq);
		if(laneId == 0){ warpBuffer[warpId] = sumSq; }
		__syncthreads();
		if(warpId == 0){
			float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
			blockSum = RMSWarpReduceSum(blockSum);
			if(laneId == 0){
				const int statIdx = n*HW + hw;
				sharedInvRms = rsqrtf(blockSum/fmaxf(static_cast<float>(C), 1.0f) + epsilon);
				invRms[statIdx] = sharedInvRms;
			}
		}
		__syncthreads();
		for(int c = tid; c < C; c += blockDim.x){
			const int idx = base + c*HW;
			y[idx] = __float2half(__half2float(x[idx])*sharedInvRms*__ldg(gamma + c));
		}
		__syncthreads();
	}
}
void RMSNormForward(__half* y, const __half* x, const float* gamma, float* invRms, int N, int C, int HW, bool spatialMode, float epsilon){
	if(!y || !x || !gamma || !invRms){
		fprintf(stderr, "RMSNormForward: Null pointer input\n");
		return;
	}
	if(N <= 0 || C <= 0 || HW <= 0 || epsilon <= 0.0f){
		fprintf(stderr, "RMSNormForward: Invalid arguments N=%d, C=%d, HW=%d, epsilon=%f\n", N, C, HW, epsilon);
		return;
	}
	if(spatialMode){
		const int threads = SelectRMSNormThreads(C);
		const int warpsPerBlock = (threads + 31)/32;
		const size_t smemSize = warpsPerBlock*sizeof(float);
		RMSNormForwardSpatialKernel<<<dim3(min(HW, 65535), N, 1), threads, smemSize>>>(y, x, gamma, invRms, N, C, HW, epsilon);
		checkCUDA(cudaGetLastError());
	} else{
		const int stride = C*HW;
		const int threads = SelectRMSNormThreads(stride);
		const int warpsPerBlock = (threads + 31)/32;
		const size_t smemSize = warpsPerBlock*sizeof(float);
		RMSNormForwardKernel<<<N, threads, smemSize>>>(y, x, gamma, invRms, N, C, HW, epsilon);
		checkCUDA(cudaGetLastError());
	}
}
__global__ void RMSNormGradGammaKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ invRms, float* __restrict__ dGamma, int N, int C, int HW){
	const int c = blockIdx.x;
	if(c >= C) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<float*>(smem);
	float sum = 0.0f;
	const int NHW = N*HW;
	const int stride = blockDim.x*gridDim.y;
	for(int nhw = tid + blockIdx.y*blockDim.x; nhw < NHW; nhw += stride){
		const int n = nhw/HW;
		const int hw = nhw % HW;
		const int idx = n*C*HW + c*HW + hw;
		sum = fmaf(__half2float(dy[idx]), __half2float(x[idx])*__ldg(invRms + n), sum);
	}
	sum = RMSWarpReduceSum(sum);
	if(laneId == 0){ warpBuffer[warpId] = sum; }
	__syncthreads();
	if(warpId == 0){
		float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
		blockSum = RMSWarpReduceSum(blockSum);
		if(laneId == 0){ atomicAdd(dGamma + c, blockSum); }
	}
}
__global__ void RMSNormGradGammaSpatialKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ invRms, float* __restrict__ dGamma, int N, int C, int HW){
	const int c = blockIdx.x;
	if(c >= C) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<float*>(smem);
	float sum = 0.0f;
	const int NHW = N*HW;
	const int stride = blockDim.x*gridDim.y;
	for(int nhw = tid + blockIdx.y*blockDim.x; nhw < NHW; nhw += stride){
		const int n = nhw/HW;
		const int hw = nhw % HW;
		const int idx = n*C*HW + c*HW + hw;
		sum = fmaf(__half2float(dy[idx]), __half2float(x[idx])*__ldg(invRms + nhw), sum);
	}
	sum = RMSWarpReduceSum(sum);
	if(laneId == 0){ warpBuffer[warpId] = sum; }
	__syncthreads();
	if(warpId == 0){
		float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
		blockSum = RMSWarpReduceSum(blockSum);
		if(laneId == 0){ atomicAdd(dGamma + c, blockSum); }
	}
}
__global__ void RMSNormDotKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ gamma, float* __restrict__ dot, int N, int C, int HW){
	const int n = blockIdx.x;
	if(n >= N) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<float*>(smem);
	const int stride = C*HW;
	const int base = n*stride;
	float sum = 0.0f;
	for(int i = tid; i < stride; i += blockDim.x){
		const int c = i/HW;
		const int idx = base + i;
		sum = fmaf(__half2float(dy[idx])*__ldg(gamma + c), __half2float(x[idx]), sum);
	}
	sum = RMSWarpReduceSum(sum);
	if(laneId == 0){ warpBuffer[warpId] = sum; }
	__syncthreads();
	if(warpId == 0){
		float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
		blockSum = RMSWarpReduceSum(blockSum);
		if(laneId == 0){ dot[n] = blockSum; }
	}
}
__global__ void RMSNormDotSpatialKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ gamma, float* __restrict__ dot, int N, int C, int HW){
	const int n = blockIdx.y;
	if(n >= N) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<float*>(smem);
	for(int hw = blockIdx.x; hw < HW; hw += gridDim.x){
		const int base = n*C*HW + hw;
		float sum = 0.0f;
		for(int c = tid; c < C; c += blockDim.x){
			const int idx = base + c*HW;
			sum = fmaf(__half2float(dy[idx])*__ldg(gamma + c), __half2float(x[idx]), sum);
		}
		sum = RMSWarpReduceSum(sum);
		if(laneId == 0){ warpBuffer[warpId] = sum; }
		__syncthreads();
		if(warpId == 0){
			float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
			blockSum = RMSWarpReduceSum(blockSum);
			if(laneId == 0){ dot[n*HW + hw] = blockSum; }
		}
		__syncthreads();
	}
}
__global__ void RMSNormInputGradKernel(__half* __restrict__ dx, const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ invRms, const float* __restrict__ dot, size_t total, int C, int HW){
	const int stride = C*HW;
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x)*gridDim.x){
		const int n = static_cast<int>(idx/stride);
		const int local = static_cast<int>(idx%stride);
		const int c = local/HW;
		const float inv = __ldg(invRms + n);
		const float xVal = __half2float(x[idx]);
		float dxVal = __ldg(gamma + c)*inv*__half2float(dy[idx]) - xVal*inv*inv*inv*__ldg(dot + n)/fmaxf(static_cast<float>(stride), 1.0f);
		dxVal = fmaxf(fminf(dxVal, RMS_GRAD_CLIP), -RMS_GRAD_CLIP);
		dx[idx] = __float2half(dxVal);
	}
}
__global__ void RMSNormInputGradSpatialKernel(__half* __restrict__ dx, const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ invRms, const float* __restrict__ dot, size_t total, int C, int HW){
	const int stride = C*HW;
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x)*gridDim.x){
		const int n = static_cast<int>(idx/stride);
		const int local = static_cast<int>(idx%stride);
		const int c = local/HW;
		const int hw = local % HW;
		const int statIdx = n*HW + hw;
		const float inv = __ldg(invRms + statIdx);
		const float xVal = __half2float(x[idx]);
		float dxVal = __ldg(gamma + c)*inv*__half2float(dy[idx]) - xVal*inv*inv*inv*__ldg(dot + statIdx)/fmaxf(static_cast<float>(C), 1.0f);
		dxVal = fmaxf(fminf(dxVal, RMS_GRAD_CLIP), -RMS_GRAD_CLIP);
		dx[idx] = __float2half(dxVal);
	}
}
void RMSNormBackward(__half* dx, const __half* dy, const __half* x, const float* gamma, float* dGamma, const float* invRms, void* workspace, size_t workspaceSize, int N, int C, int HW, bool spatialMode){
	if(!dx || !dy || !x || !gamma || !dGamma || !invRms || !workspace){
		fprintf(stderr, "RMSNormBackward: Null pointer input\n");
		return;
	}
	if(N <= 0 || C <= 0 || HW <= 0){
		fprintf(stderr, "RMSNormBackward: Invalid dimensions N=%d, C=%d, HW=%d\n", N, C, HW);
		return;
	}
	const size_t statsCount = spatialMode ? static_cast<size_t>(N)*static_cast<size_t>(HW) : static_cast<size_t>(N);
	const size_t requiredSize = statsCount*sizeof(float);
	if(workspaceSize < requiredSize){
		fprintf(stderr, "RMSNormBackward: Insufficient workspace (need %zu, got %zu)\n", requiredSize, workspaceSize);
		return;
	}
	auto* dot = static_cast<float*>(workspace);
	checkCUDA(cudaMemset(dGamma, 0, C*sizeof(float)));
	const int gradTpb = 256;
	const int gradWarps = (gradTpb + 31)/32;
	const size_t gradSmem = gradWarps*sizeof(float);
	const int nhw = N*HW;
	const int gradGridY = max(1, min(DivCeil(nhw, gradTpb*16), 256));
	if(spatialMode){
		RMSNormGradGammaSpatialKernel<<<dim3(C, gradGridY, 1), gradTpb, gradSmem>>>(dy, x, invRms, dGamma, N, C, HW);
		checkCUDA(cudaGetLastError());
		const int dotTpb = SelectRMSNormThreads(C);
		const int dotWarps = (dotTpb + 31)/32;
		const size_t dotSmem = dotWarps*sizeof(float);
		RMSNormDotSpatialKernel<<<dim3(min(HW, 65535), N, 1), dotTpb, dotSmem>>>(dy, x, gamma, dot, N, C, HW);
		checkCUDA(cudaGetLastError());
		size_t blocks, tpb = 256;
		const size_t total = static_cast<size_t>(N)*C*HW;
		GetLaunchConfigGridStride(total, blocks, tpb);
		RMSNormInputGradSpatialKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(dx, dy, x, gamma, invRms, dot, total, C, HW);
		checkCUDA(cudaGetLastError());
	} else{
		RMSNormGradGammaKernel<<<dim3(C, gradGridY, 1), gradTpb, gradSmem>>>(dy, x, invRms, dGamma, N, C, HW);
		checkCUDA(cudaGetLastError());
		const int stride = C*HW;
		const int dotTpb = SelectRMSNormThreads(stride);
		const int dotWarps = (dotTpb + 31)/32;
		const size_t dotSmem = dotWarps*sizeof(float);
		RMSNormDotKernel<<<N, dotTpb, dotSmem>>>(dy, x, gamma, dot, N, C, HW);
		checkCUDA(cudaGetLastError());
		size_t blocks, tpb = 256;
		const size_t total = static_cast<size_t>(N)*C*HW;
		GetLaunchConfigGridStride(total, blocks, tpb);
		RMSNormInputGradKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(dx, dy, x, gamma, invRms, dot, total, C, HW);
		checkCUDA(cudaGetLastError());
	}
}
size_t RMSNormBackwardWorkspaceSize(int N, int HW, bool spatialMode){
	const size_t statsCount = spatialMode ? static_cast<size_t>(N)*static_cast<size_t>(HW) : static_cast<size_t>(N);
	return statsCount*sizeof(float);
}
