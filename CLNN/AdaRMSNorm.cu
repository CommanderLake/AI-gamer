#define __CUDACC__
#include "CuCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#define ADA_RMS_GRAD_CLIP 5.0f
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
__device__ __forceinline__ int AdaRMSModulationRow(const int n, const int modulationBatchSize, const int rowsPerModulation){
	if(modulationBatchSize <= 0) return 0;
	int modulationRow = n/rowsPerModulation;
	if(modulationRow >= modulationBatchSize){ modulationRow = modulationBatchSize - 1; }
	return modulationRow;
}
__global__ void AdaRMSNormGradGammaKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ invRms, const __half* __restrict__ scale, const __half* __restrict__ gate, float* __restrict__ dGamma, int N, int C, int HW, int modulationBatchSize, int rowsPerModulation){
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
		const int modRow = AdaRMSModulationRow(n, modulationBatchSize, rowsPerModulation);
		const int modIdx = modRow*C + c;
		const float modScale = scale ? __half2float(scale[modIdx]) : 0.0f;
		const float modGate = gate ? __half2float(gate[modIdx]) : 1.0f;
		sum = fmaf(__half2float(dy[idx])*modGate*(1.0f + modScale), __half2float(x[idx])*__ldg(invRms + n), sum);
	}
	sum = AdaRMSWarpReduceSum(sum);
	if(laneId == 0){ warpBuffer[warpId] = sum; }
	__syncthreads();
	if(warpId == 0){
		float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
		blockSum = AdaRMSWarpReduceSum(blockSum);
		if(laneId == 0){ atomicAdd(dGamma + c, blockSum); }
	}
}
__global__ void AdaRMSNormModulationGradKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ invRms, const __half* __restrict__ scale, const __half* __restrict__ shift, const __half* __restrict__ gate, __half* __restrict__ gradScale, __half* __restrict__ gradShift, __half* __restrict__ gradGate, int N, int C, int HW, int modulationBatchSize, int rowsPerModulation){
	const int c = blockIdx.x;
	const int modRow = blockIdx.y;
	if(c >= C || modRow >= modulationBatchSize) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpScale = reinterpret_cast<float*>(smem);
	float* warpShift = warpScale + warpsPerBlock;
	float* warpGate = warpShift + warpsPerBlock;
	float scaleSum = 0.0f;
	float shiftSum = 0.0f;
	float gateSum = 0.0f;
	const int nStart = modRow*rowsPerModulation;
	int nEnd = nStart + rowsPerModulation;
	if(nEnd > N){ nEnd = N; }
	const int modIdx = modRow*C + c;
	const float modScale = scale ? __half2float(scale[modIdx]) : 0.0f;
	const float modShift = shift ? __half2float(shift[modIdx]) : 0.0f;
	const float modGate = gate ? __half2float(gate[modIdx]) : 1.0f;
	for(int local = tid; local < (nEnd - nStart)*HW; local += blockDim.x){
		const int n = nStart + local/HW;
		const int hw = local % HW;
		const int idx = n*C*HW + c*HW + hw;
		const float dyVal = __half2float(dy[idx]);
		const float norm = __half2float(x[idx])*__ldg(invRms + n)*__ldg(gamma + c);
		if(gradScale){ scaleSum = fmaf(dyVal*modGate, norm, scaleSum); }
		if(gradShift){ shiftSum += dyVal*modGate; }
		if(gradGate){ gateSum = fmaf(dyVal, norm*(1.0f + modScale) + modShift, gateSum); }
	}
	scaleSum = AdaRMSWarpReduceSum(scaleSum);
	shiftSum = AdaRMSWarpReduceSum(shiftSum);
	gateSum = AdaRMSWarpReduceSum(gateSum);
	if(laneId == 0){
		warpScale[warpId] = scaleSum;
		warpShift[warpId] = shiftSum;
		warpGate[warpId] = gateSum;
	}
	__syncthreads();
	if(warpId == 0){
		float blockScale = laneId < warpsPerBlock ? warpScale[laneId] : 0.0f;
		float blockShift = laneId < warpsPerBlock ? warpShift[laneId] : 0.0f;
		float blockGate = laneId < warpsPerBlock ? warpGate[laneId] : 0.0f;
		blockScale = AdaRMSWarpReduceSum(blockScale);
		blockShift = AdaRMSWarpReduceSum(blockShift);
		blockGate = AdaRMSWarpReduceSum(blockGate);
		if(laneId == 0){
			if(gradScale){ gradScale[modIdx] = __float2half(blockScale); }
			if(gradShift){ gradShift[modIdx] = __float2half(blockShift); }
			if(gradGate){ gradGate[modIdx] = __float2half(blockGate); }
		}
	}
}
__global__ void AdaRMSNormDotKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ gamma, const __half* __restrict__ scale, const __half* __restrict__ gate, float* __restrict__ dot, int N, int C, int HW, int modulationBatchSize, int rowsPerModulation){
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
	const int modRow = AdaRMSModulationRow(n, modulationBatchSize, rowsPerModulation);
	float sum = 0.0f;
	for(int i = tid; i < stride; i += blockDim.x){
		const int c = i/HW;
		const int modIdx = modRow*C + c;
		const float modScale = scale ? __half2float(scale[modIdx]) : 0.0f;
		const float modGate = gate ? __half2float(gate[modIdx]) : 1.0f;
		const int idx = base + i;
		const float dNorm = __half2float(dy[idx])*modGate*(1.0f + modScale);
		sum = fmaf(dNorm*__ldg(gamma + c), __half2float(x[idx]), sum);
	}
	sum = AdaRMSWarpReduceSum(sum);
	if(laneId == 0){ warpBuffer[warpId] = sum; }
	__syncthreads();
	if(warpId == 0){
		float blockSum = laneId < warpsPerBlock ? warpBuffer[laneId] : 0.0f;
		blockSum = AdaRMSWarpReduceSum(blockSum);
		if(laneId == 0){ dot[n] = blockSum; }
	}
}
__global__ void AdaRMSNormInputGradKernel(__half* __restrict__ dx, const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ gamma, const float* __restrict__ invRms, const __half* __restrict__ scale, const __half* __restrict__ gate, const float* __restrict__ dot, size_t total, int C, int HW, int modulationBatchSize, int rowsPerModulation){
	const int stride = C*HW;
	for(size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += static_cast<size_t>(blockDim.x)*gridDim.x){
		const int n = static_cast<int>(idx/stride);
		const int local = static_cast<int>(idx%stride);
		const int c = local/HW;
		const int modRow = AdaRMSModulationRow(n, modulationBatchSize, rowsPerModulation);
		const int modIdx = modRow*C + c;
		const float modScale = scale ? __half2float(scale[modIdx]) : 0.0f;
		const float modGate = gate ? __half2float(gate[modIdx]) : 1.0f;
		const float dNorm = __half2float(dy[idx])*modGate*(1.0f + modScale);
		const float inv = __ldg(invRms + n);
		const float xVal = __half2float(x[idx]);
		float dxVal = __ldg(gamma + c)*inv*dNorm - xVal*inv*inv*inv*__ldg(dot + n)/fmaxf(static_cast<float>(stride), 1.0f);
		dxVal = fmaxf(fminf(dxVal, ADA_RMS_GRAD_CLIP), -ADA_RMS_GRAD_CLIP);
		dx[idx] = __float2half(dxVal);
	}
}
void AdaRMSNormBackward(__half* dx, const __half* dy, const __half* x, const float* gamma, float* dGamma, const float* invRms, const __half* scale, const __half* shift, const __half* gate, __half* gradScale, __half* gradShift, __half* gradGate, void* workspace, size_t workspaceSize, int N, int C, int HW, int modulationBatchSize, int rowsPerModulation){
	if(!dx || !dy || !x || !gamma || !dGamma || !invRms || !workspace){
		fprintf(stderr, "AdaRMSNormBackward: Null pointer input\n");
		return;
	}
	if(N <= 0 || C <= 0 || HW <= 0 || rowsPerModulation <= 0){
		fprintf(stderr, "AdaRMSNormBackward: Invalid dimensions N=%d, C=%d, HW=%d, rowsPerModulation=%d\n", N, C, HW, rowsPerModulation);
		return;
	}
	if((scale || shift || gate || gradScale || gradShift || gradGate) && modulationBatchSize <= 0){
		fprintf(stderr, "AdaRMSNormBackward: Modulation tensors require a positive modulation batch size\n");
		return;
	}
	const size_t requiredSize = static_cast<size_t>(N)*sizeof(float);
	if(workspaceSize < requiredSize){
		fprintf(stderr, "AdaRMSNormBackward: Insufficient workspace (need %zu, got %zu)\n", requiredSize, workspaceSize);
		return;
	}
	auto* dot = static_cast<float*>(workspace);
	checkCUDA(cudaMemset(dGamma, 0, C*sizeof(float)));
	const int gradTpb = 256;
	const int gradWarps = (gradTpb + 31)/32;
	const size_t gradSmem = gradWarps*sizeof(float);
	const int nhw = N*HW;
	const int gradGridY = max(1, min(DivCeil(nhw, gradTpb*16), 256));
	AdaRMSNormGradGammaKernel<<<dim3(C, gradGridY, 1), gradTpb, gradSmem>>>(dy, x, invRms, scale, gate, dGamma, N, C, HW, modulationBatchSize, rowsPerModulation);
	checkCUDA(cudaGetLastError());
	if(modulationBatchSize > 0 && (gradScale || gradShift || gradGate)){
		const size_t modSmem = 3*gradWarps*sizeof(float);
		AdaRMSNormModulationGradKernel<<<dim3(C, modulationBatchSize, 1), gradTpb, modSmem>>>(dy, x, gamma, invRms, scale, shift, gate, gradScale, gradShift, gradGate, N, C, HW, modulationBatchSize, rowsPerModulation);
		checkCUDA(cudaGetLastError());
	}
	const int stride = C*HW;
	const int dotTpb = SelectAdaRMSNormThreads(stride);
	const int dotWarps = (dotTpb + 31)/32;
	const size_t dotSmem = dotWarps*sizeof(float);
	AdaRMSNormDotKernel<<<N, dotTpb, dotSmem>>>(dy, x, gamma, scale, gate, dot, N, C, HW, modulationBatchSize, rowsPerModulation);
	checkCUDA(cudaGetLastError());
	size_t blocks, tpb = 256;
	const size_t total = static_cast<size_t>(N)*C*HW;
	GetLaunchConfigGridStride(total, blocks, tpb);
	AdaRMSNormInputGradKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(dx, dy, x, gamma, invRms, scale, gate, dot, total, C, HW, modulationBatchSize, rowsPerModulation);
	checkCUDA(cudaGetLastError());
}
size_t AdaRMSNormBackwardWorkspaceSize(int N){
	return static_cast<size_t>(N)*sizeof(float);
}
