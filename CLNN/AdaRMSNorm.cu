#define __CUDACC__
#include "CuCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
__device__ __forceinline__ float AdaRMSWarpReduceSum(float value){
#pragma unroll
	for(int offset = 16; offset > 0; offset >>= 1){ value += __shfl_down_sync(0xFFFFFFFF, value, offset); }
	return value;
}
int SelectAdaRMSNormThreads(const int elements){
	int threads = 32;
	while(threads < elements && threads < 512) threads <<= 1;
	if(elements < 32){ threads = 32; }
	return threads;
}
__global__ void AdaRMSNormForwardKernel(__half* __restrict__ y, const __half* __restrict__ x, const float* __restrict__ gamma, const __half* __restrict__ scale, const __half* __restrict__ shift, const __half* __restrict__ gate, float* __restrict__ invRms, int N, int C, int HW, int modulationBatchSize, int rowsPerModulation, float epsilon){
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
	sumSq = AdaRMSWarpReduceSum(sumSq);
	if(laneId == 0){ warpBuffer[warpId] = sumSq; }
	__syncthreads();
	if(warpId == 0){
		float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
		blockSum = AdaRMSWarpReduceSum(blockSum);
		if(laneId == 0){
			sharedInvRms = rsqrtf(blockSum/fmaxf(static_cast<float>(stride), 1.0f) + epsilon);
			invRms[n] = sharedInvRms;
		}
	}
	__syncthreads();
	int modulationRow = 0;
	if(modulationBatchSize > 0){
		modulationRow = n/rowsPerModulation;
		if(modulationRow >= modulationBatchSize){ modulationRow = modulationBatchSize - 1; }
	}
	for(int i = tid; i < stride; i += blockDim.x){
		const int c = i/HW;
		const int modIdx = modulationRow*C + c;
		const float modScale = scale ? __half2float(scale[modIdx]) : 0.0f;
		const float modShift = shift ? __half2float(shift[modIdx]) : 0.0f;
		const float modGate = gate ? __half2float(gate[modIdx]) : 1.0f;
		const float norm = __half2float(x[base + i])*sharedInvRms*__ldg(gamma + c);
		y[base + i] = __float2half((norm*(1.0f + modScale) + modShift)*modGate);
	}
}
void AdaRMSNormForward(__half* y, const __half* x, const float* gamma, const __half* scale, const __half* shift, const __half* gate, float* invRms, int N, int C, int HW, int modulationBatchSize, int rowsPerModulation, float epsilon){
	if(!y || !x || !gamma || !invRms){
		fprintf(stderr, "AdaRMSNormForward: Null pointer input\n");
		return;
	}
	if(N <= 0 || C <= 0 || HW <= 0 || epsilon <= 0.0f || rowsPerModulation <= 0){
		fprintf(stderr, "AdaRMSNormForward: Invalid arguments N=%d, C=%d, HW=%d, rowsPerModulation=%d, epsilon=%f\n", N, C, HW, rowsPerModulation, epsilon);
		return;
	}
	if((scale || shift || gate) && modulationBatchSize <= 0){
		fprintf(stderr, "AdaRMSNormForward: Modulation tensors require a positive modulation batch size\n");
		return;
	}
	const int stride = C*HW;
	const int threads = SelectAdaRMSNormThreads(stride);
	const int warpsPerBlock = (threads + 31)/32;
	const size_t smemSize = warpsPerBlock*sizeof(float);
	AdaRMSNormForwardKernel<<<N, threads, smemSize>>>(y, x, gamma, scale, shift, gate, invRms, N, C, HW, modulationBatchSize, rowsPerModulation, epsilon);
	checkCUDA(cudaGetLastError());
}
