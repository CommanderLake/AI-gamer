#define __CUDACC__
#include "CuCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#define GROUP_NORM_GRAD_CLIP 5.0f
struct GroupNormPair{
	float x;
	float y;
};
__device__ __forceinline__ GroupNormPair GroupNormWarpReducePair(GroupNormPair value){
#pragma unroll
	for(int offset = 16; offset > 0; offset >>= 1){
		value.x += __shfl_down_sync(0xFFFFFFFF, value.x, offset);
		value.y += __shfl_down_sync(0xFFFFFFFF, value.y, offset);
	}
	return value;
}
int SelectGroupNormThreads(const int elements){
	int threads = 32;
	while(threads < elements && threads < 512) threads <<= 1;
	if(elements < 32){ threads = 32; }
	return threads;
}
__global__ void GroupNormForwardKernel(__half* __restrict__ y, const __half* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ beta, float* __restrict__ mean, float* __restrict__ invStd, int N, int C, int HW, int groups, float epsilon){
	const int statIdx = blockIdx.x;
	const int statsCount = N*groups;
	if(statIdx >= statsCount) return;
	const int n = statIdx/groups;
	const int group = statIdx % groups;
	const int channelsPerGroup = C/groups;
	const int groupElements = channelsPerGroup*HW;
	const int channelBase = group*channelsPerGroup;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<GroupNormPair*>(smem);
	__shared__ float sharedMean;
	__shared__ float sharedInvStd;
	GroupNormPair sums{0.0f, 0.0f};
	for(int i = tid; i < groupElements; i += blockDim.x){
		const int c = channelBase + i/HW;
		const int hw = i % HW;
		const int idx = n*C*HW + c*HW + hw;
		const float v = __half2float(x[idx]);
		sums.x += v;
		sums.y = fmaf(v, v, sums.y);
	}
	sums = GroupNormWarpReducePair(sums);
	if(laneId == 0){ warpBuffer[warpId] = sums; }
	__syncthreads();
	if(warpId == 0){
		GroupNormPair blockSums{0.0f, 0.0f};
		if(laneId < warpsPerBlock){ blockSums = warpBuffer[laneId]; }
		blockSums = GroupNormWarpReducePair(blockSums);
		if(laneId == 0){
			const float invM = 1.0f/fmaxf(static_cast<float>(groupElements), 1.0f);
			sharedMean = blockSums.x*invM;
			const float variance = fmaxf(blockSums.y*invM - sharedMean*sharedMean, 0.0f);
			sharedInvStd = rsqrtf(variance + epsilon);
			mean[statIdx] = sharedMean;
			invStd[statIdx] = sharedInvStd;
		}
	}
	__syncthreads();
	for(int i = tid; i < groupElements; i += blockDim.x){
		const int c = channelBase + i/HW;
		const int hw = i % HW;
		const int idx = n*C*HW + c*HW + hw;
		const float norm = (__half2float(x[idx]) - sharedMean)*sharedInvStd;
		y[idx] = __float2half(fmaf(norm, __ldg(gamma + c), __ldg(beta + c)));
	}
}
void GroupNormForward(__half* y, const __half* x, const float* gamma, const float* beta, float* mean, float* invStd, int N, int C, int HW, int groups, float epsilon){
	if(!y || !x || !gamma || !beta || !mean || !invStd){
		fprintf(stderr, "GroupNormForward: Null pointer input\n");
		return;
	}
	if(N <= 0 || C <= 0 || HW <= 0 || groups <= 0 || C%groups != 0 || epsilon <= 0.0f){
		fprintf(stderr, "GroupNormForward: Invalid arguments N=%d, C=%d, HW=%d, groups=%d, epsilon=%f\n", N, C, HW, groups, epsilon);
		return;
	}
	const int groupElements = C/groups*HW;
	const int threads = SelectGroupNormThreads(groupElements);
	const int warpsPerBlock = (threads + 31)/32;
	const size_t smemSize = warpsPerBlock*sizeof(GroupNormPair);
	GroupNormForwardKernel<<<N*groups, threads, smemSize>>>(y, x, gamma, beta, mean, invStd, N, C, HW, groups, epsilon);
	checkCUDA(cudaGetLastError());
}
__global__ void GroupNormParamGradKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ mean, const float* __restrict__ invStd, float* __restrict__ dGamma, float* __restrict__ dBeta, int N, int C, int HW, int groups){
	const int c = blockIdx.x;
	if(c >= C) return;
	const int channelsPerGroup = C/groups;
	const int group = c/channelsPerGroup;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<GroupNormPair*>(smem);
	GroupNormPair sums{0.0f, 0.0f};
	const int NHW = N*HW;
	const int stride = blockDim.x*gridDim.y;
	for(int nhw = tid + blockIdx.y*blockDim.x; nhw < NHW; nhw += stride){
		const int n = nhw/HW;
		const int hw = nhw % HW;
		const int idx = n*C*HW + c*HW + hw;
		const int statIdx = n*groups + group;
		const float dyv = __half2float(dy[idx]);
		const float xhat = (__half2float(x[idx]) - __ldg(mean + statIdx))*__ldg(invStd + statIdx);
		sums.x = fmaf(dyv, xhat, sums.x);
		sums.y += dyv;
	}
	sums = GroupNormWarpReducePair(sums);
	if(laneId == 0){ warpBuffer[warpId] = sums; }
	__syncthreads();
	if(warpId == 0){
		GroupNormPair blockSums{0.0f, 0.0f};
		if(laneId < warpsPerBlock){ blockSums = warpBuffer[laneId]; }
		blockSums = GroupNormWarpReducePair(blockSums);
		if(laneId == 0){
			atomicAdd(dGamma + c, blockSums.x);
			atomicAdd(dBeta + c, blockSums.y);
		}
	}
}
__global__ void GroupNormStatsGradKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ mean, const float* __restrict__ invStd, float* __restrict__ d1, float* __restrict__ d2, int N, int C, int HW, int groups){
	const int statIdx = blockIdx.x;
	const int statsCount = N*groups;
	if(statIdx >= statsCount) return;
	const int n = statIdx/groups;
	const int group = statIdx % groups;
	const int channelsPerGroup = C/groups;
	const int groupElements = channelsPerGroup*HW;
	const int channelBase = group*channelsPerGroup;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<GroupNormPair*>(smem);
	GroupNormPair sums{0.0f, 0.0f};
	const float m = __ldg(mean + statIdx);
	const float inv = __ldg(invStd + statIdx);
	for(int i = tid; i < groupElements; i += blockDim.x){
		const int c = channelBase + i/HW;
		const int hw = i % HW;
		const int idx = n*C*HW + c*HW + hw;
		const float dyGamma = __half2float(dy[idx])*__ldg(gamma + c);
		const float xhat = (__half2float(x[idx]) - m)*inv;
		sums.x += dyGamma;
		sums.y = fmaf(dyGamma, xhat, sums.y);
	}
	sums = GroupNormWarpReducePair(sums);
	if(laneId == 0){ warpBuffer[warpId] = sums; }
	__syncthreads();
	if(warpId == 0){
		GroupNormPair blockSums{0.0f, 0.0f};
		if(laneId < warpsPerBlock){ blockSums = warpBuffer[laneId]; }
		blockSums = GroupNormWarpReducePair(blockSums);
		if(laneId == 0){
			d1[statIdx] = blockSums.x;
			d2[statIdx] = blockSums.y;
		}
	}
}
__global__ void GroupNormInputGradKernel(__half* __restrict__ dx, const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ mean, const float* __restrict__ invStd, const float* __restrict__ d1, const float* __restrict__ d2, size_t total, int C, int HW, int groups){
	const int stride = C*HW;
	const int channelsPerGroup = C/groups;
	const int groupElements = channelsPerGroup*HW;
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x)*gridDim.x){
		const int n = static_cast<int>(idx/stride);
		const int local = static_cast<int>(idx%stride);
		const int c = local/HW;
		const int group = c/channelsPerGroup;
		const int statIdx = n*groups + group;
		const float inv = __ldg(invStd + statIdx);
		const float xhat = (__half2float(x[idx]) - __ldg(mean + statIdx))*inv;
		const float dyGamma = __half2float(dy[idx])*__ldg(gamma + c);
		const float invM = 1.0f/fmaxf(static_cast<float>(groupElements), 1.0f);
		float dxv = inv*(dyGamma - __ldg(d1 + statIdx)*invM - xhat*__ldg(d2 + statIdx)*invM);
		dxv = fmaxf(fminf(dxv, GROUP_NORM_GRAD_CLIP), -GROUP_NORM_GRAD_CLIP);
		dx[idx] = __float2half(dxv);
	}
}
void GroupNormBackward(__half* dx, const __half* dy, const __half* x, const float* gamma, float* dGamma, float* dBeta, const float* mean, const float* invStd, void* workspace, size_t workspaceSize, int N, int C, int HW, int groups){
	if(!dx || !dy || !x || !gamma || !dGamma || !dBeta || !mean || !invStd || !workspace){
		fprintf(stderr, "GroupNormBackward: Null pointer input\n");
		return;
	}
	if(N <= 0 || C <= 0 || HW <= 0 || groups <= 0 || C%groups != 0){
		fprintf(stderr, "GroupNormBackward: Invalid dimensions N=%d, C=%d, HW=%d, groups=%d\n", N, C, HW, groups);
		return;
	}
	const size_t statsCount = static_cast<size_t>(N)*groups;
	const size_t requiredSize = 2*statsCount*sizeof(float);
	if(workspaceSize < requiredSize){
		fprintf(stderr, "GroupNormBackward: Insufficient workspace (need %zu, got %zu)\n", requiredSize, workspaceSize);
		return;
	}
	auto* d1 = static_cast<float*>(workspace);
	float* d2 = d1 + statsCount;
	checkCUDA(cudaMemset(dGamma, 0, C*sizeof(float)));
	checkCUDA(cudaMemset(dBeta, 0, C*sizeof(float)));
	const int paramTpb = 256;
	const int paramWarps = (paramTpb + 31)/32;
	const size_t paramSmem = paramWarps*sizeof(GroupNormPair);
	const int nhw = N*HW;
	const int paramGridY = max(1, min(DivCeil(nhw, paramTpb*16), 256));
	GroupNormParamGradKernel<<<dim3(C, paramGridY, 1), paramTpb, paramSmem>>>(dy, x, mean, invStd, dGamma, dBeta, N, C, HW, groups);
	checkCUDA(cudaGetLastError());
	const int groupElements = C/groups*HW;
	const int statsTpb = SelectGroupNormThreads(groupElements);
	const int statsWarps = (statsTpb + 31)/32;
	const size_t statsSmem = statsWarps*sizeof(GroupNormPair);
	GroupNormStatsGradKernel<<<N*groups, statsTpb, statsSmem>>>(dy, x, gamma, mean, invStd, d1, d2, N, C, HW, groups);
	checkCUDA(cudaGetLastError());
	size_t blocks, tpb = 256;
	const size_t total = static_cast<size_t>(N)*C*HW;
	GetLaunchConfigGridStride(total, blocks, tpb);
	GroupNormInputGradKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(dx, dy, x, gamma, mean, invStd, d1, d2, total, C, HW, groups);
	checkCUDA(cudaGetLastError());
}
size_t GroupNormBackwardWorkspaceSize(int N, int groups){
	return 2*static_cast<size_t>(N)*groups*sizeof(float);
}
