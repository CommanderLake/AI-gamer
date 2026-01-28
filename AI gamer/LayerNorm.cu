#define __CUDACC__
#include "CuCommon.cuh"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cassert>
// Maximum gradient clipping value
#define LN_GRAD_CLIP 5.0f
namespace{
	struct WelfordData{
		float mean;
		float m2;
		float count;
	};
	struct PairData{
		float x;
		float y;
	};
	__device__ __forceinline__ WelfordData WelfordUpdate(WelfordData data, float x){
		data.count += 1.0f;
		const float delta = x - data.mean;
		data.mean += delta/data.count;
		const float delta2 = x - data.mean;
		data.m2 += delta*delta2;
		return data;
	}
	__device__ __forceinline__ WelfordData WelfordCombine(WelfordData a, const WelfordData& b){
		if(b.count == 0.0f){ return a; }
		if(a.count == 0.0f){ return b; }
		const float delta = b.mean - a.mean;
		const float count = a.count + b.count;
		a.mean += delta*(b.count/count);
		a.m2 += b.m2 + delta*delta*(a.count*b.count)/count;
		a.count = count;
		return a;
	}
	__device__ __forceinline__ WelfordData WarpReduceWelford(WelfordData value){
#pragma unroll
		for(int offset = 16; offset > 0; offset >>= 1){
			WelfordData other;
			other.mean = __shfl_down_sync(0xFFFFFFFF, value.mean, offset);
			other.m2 = __shfl_down_sync(0xFFFFFFFF, value.m2, offset);
			other.count = __shfl_down_sync(0xFFFFFFFF, value.count, offset);
			value = WelfordCombine(value, other);
		}
		return value;
	}
	__device__ __forceinline__ PairData WarpReducePair(PairData value){
#pragma unroll
		for(int offset = 16; offset > 0; offset >>= 1){
			value.x += __shfl_down_sync(0xFFFFFFFF, value.x, offset);
			value.y += __shfl_down_sync(0xFFFFFFFF, value.y, offset);
		}
		return value;
	}
	int SelectLayerNormThreads(const int elements){
		int threads = 32;
		while(threads < elements && threads < 512) threads <<= 1;
		if(elements < 32){ threads = 32; }
		return threads;
	}
}
__global__ void ComputeMeanVarianceKernel(const __half* __restrict__ x, float* __restrict__ mean, float* __restrict__ var, int N, int C, int HW){
	const int n = blockIdx.x;
	if(n >= N) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<WelfordData*>(smem);
	const int stride = C*HW;
	const __half* xn = x + n*stride;
	WelfordData data{0.0f, 0.0f, 0.0f};
	for(int i = tid; i < stride; i += blockDim.x){ data = WelfordUpdate(data, __half2float(xn[i])); }
	data = WarpReduceWelford(data);
	if(laneId == 0){ warpBuffer[warpId] = data; }
	__syncthreads();
	if(warpId == 0){
		WelfordData blockData{0.0f, 0.0f, 0.0f};
		if(laneId < warpsPerBlock){ blockData = warpBuffer[laneId]; }
		blockData = WarpReduceWelford(blockData);
		if(laneId == 0){
			mean[n] = blockData.mean;
			const float varValue = blockData.count > 0.0f ? blockData.m2/blockData.count : 0.0f;
			var[n] = fmaxf(varValue, 0.0f);
		}
	}
}
__global__ void LayerNormForwardKernel(__half* __restrict__ y, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b, const float* __restrict__ mean, const float* __restrict__ var, int N, int C, int HW){
	const int n = blockIdx.y;
	if(n >= N) return;
	const int stride = C*HW;
	const int base = n*stride;
	const float m = mean[n];
	const float varSafe = fmaxf(var[n], 0.0f);
	const float invStd = rsqrtf(varSafe + EPSILON_F);
	const int totalTiles = (stride + blockDim.x - 1)/blockDim.x;
	for(int tile = blockIdx.x; tile < totalTiles; tile += gridDim.x){
		const int localIdx = tile*blockDim.x + threadIdx.x;
		if(localIdx < stride){
			const int c = localIdx/HW;
			const int idx = base + localIdx;
			const float v = __half2float(x[idx]);
			const float norm = (v - m)*invStd;
			// Use texture cache for parameter loads
			const float gamma = __ldg(g + c);
			const float beta = __ldg(b + c);
			y[idx] = __float2half(fmaf(norm, gamma, beta));
		}
	}
}
__global__ void ComputeMeanVarianceSpatialKernel(const __half* __restrict__ x, float* __restrict__ mean, float* __restrict__ var, int N, int C, int HW){
	const int n = blockIdx.y;
	if(n >= N) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<WelfordData*>(smem);
	for(int hw = blockIdx.x; hw < HW; hw += gridDim.x){
		WelfordData data{0.0f, 0.0f, 0.0f};
		const int base = n*C*HW + hw;
		for(int c = tid; c < C; c += blockDim.x){
			const int idx = base + c*HW;
			data = WelfordUpdate(data, __half2float(x[idx]));
		}
		data = WarpReduceWelford(data);
		if(laneId == 0){ warpBuffer[warpId] = data; }
		__syncthreads();
		if(warpId == 0){
			WelfordData blockData{0.0f, 0.0f, 0.0f};
			if(laneId < warpsPerBlock){ blockData = warpBuffer[laneId]; }
			blockData = WarpReduceWelford(blockData);
			if(laneId == 0){
				const int nhw = n*HW + hw;
				mean[nhw] = blockData.mean;
				const float varValue = blockData.count > 0.0f ? blockData.m2/blockData.count : 0.0f;
				var[nhw] = fmaxf(varValue, 0.0f);
			}
		}
		__syncthreads();
	}
}

__global__ void LayerNormForwardSpatialKernel(__half* __restrict__ y, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b, const float* __restrict__ mean, const float* __restrict__ var, int N, int C, int HW){
	const int n = blockIdx.y;
	if(n >= N) return;
	for(int hw = blockIdx.x; hw < HW; hw += gridDim.x){
		const int nhw = n*HW + hw;
		const int base = n*C*HW + hw;
		const float m = mean[nhw];
		const float varSafe = fmaxf(var[nhw], 0.0f);
		const float invStd = rsqrtf(varSafe + EPSILON_F);
		for(int c = threadIdx.x; c < C; c += blockDim.x){
			const int idx = base + c*HW;
			const float v = __half2float(x[idx]);
			const float norm = (v - m)*invStd;
			const float gamma = __ldg(g + c);
			const float beta = __ldg(b + c);
			y[idx] = __float2half(fmaf(norm, gamma, beta));
		}
	}
}
void LayerNormForward(__half* y, const __half* x, const float* g, const float* b, float* mean, float* var, int N, int C, int HW, bool spatialMode){
	// Validate inputs
	if(!y || !x || !g || !b || !mean || !var){
		fprintf(stderr, "LayerNormForward: Null pointer input\n");
		return;
	}
	if(N <= 0 || C <= 0 || HW <= 0){
		fprintf(stderr, "LayerNormForward: Invalid dimensions N=%d, C=%d, HW=%d\n", N, C, HW);
		return;
	}
	const int stride = C*HW;
	if(spatialMode){
		const int threads = SelectLayerNormThreads(C);
		const int warpsPerBlock = (threads + 31)/32;
		const size_t smemSize = warpsPerBlock*sizeof(WelfordData);
		int maxSmem;
		cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
		if(smemSize > maxSmem){
			fprintf(stderr, "LayerNormForward: Required shared memory %zu exceeds limit %d\n", smemSize, maxSmem);
			return;
		}
		const int gridX = min(HW, 65535);
		dim3 grid(gridX, N, 1);
		ComputeMeanVarianceSpatialKernel<<<grid, threads, smemSize>>>(x, mean, var, N, C, HW);
		checkCUDA(cudaGetLastError());
		LayerNormForwardSpatialKernel<<<grid, threads>>>(y, x, g, b, mean, var, N, C, HW);
		checkCUDA(cudaGetLastError());
	} else{
		const int threads = SelectLayerNormThreads(stride);
		const int warpsPerBlock = (threads + 31)/32;
		const size_t smemSize = warpsPerBlock*sizeof(WelfordData);
		int maxSmem;
		cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
		if(smemSize > maxSmem){
			fprintf(stderr, "LayerNormForward: Required shared memory %zu exceeds limit %d\n", smemSize, maxSmem);
			return;
		}
		ComputeMeanVarianceKernel<<<N, threads, smemSize>>>(x, mean, var, N, C, HW);
		checkCUDA(cudaGetLastError());
		const int tiles = (stride + threads - 1)/threads;
		int gridX = min(tiles, 65535);
		dim3 grid(gridX, N, 1);
		LayerNormForwardKernel<<<grid, threads>>>(y, x, g, b, mean, var, N, C, HW);
		checkCUDA(cudaGetLastError());
	}
}
__global__ void GradGammaBetaKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ mean, const float* __restrict__ var, float* __restrict__ dG, float* __restrict__ dB, int N, int C, int HW){
	const int cid = blockIdx.x;
	if(cid >= C) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<PairData*>(smem);
	PairData threadData{0.0f, 0.0f};
	// Accumulate across batch dimension
	for(int n = blockIdx.y; n < N; n += gridDim.y){
		const float varSafe = fmaxf(__ldg(var + n), 0.0f);
		const float invStd = rsqrtf(varSafe + EPSILON_F);
		const float m = __ldg(mean + n);
		const int base = n*C*HW + cid*HW;
		for(int i = tid; i < HW; i += blockDim.x){
			const int idx = base + i;
			const float dyv = __half2float(dy[idx]);
			const float xv = __half2float(x[idx]);
			const float xnorm = (xv - m)*invStd;
			threadData.x = fmaf(xnorm, dyv, threadData.x);
			threadData.y += dyv;
		}
	}
	// Warp reduction
	threadData = WarpReducePair(threadData);
	if(laneId == 0){ warpBuffer[warpId] = threadData; }
	__syncthreads();
	// Final reduction in first warp
	if(warpId == 0){
		PairData blockData{0.0f, 0.0f};
		if(laneId < warpsPerBlock){ blockData = warpBuffer[laneId]; }
		blockData = WarpReducePair(blockData);
		if(laneId == 0){
			atomicAdd(dG + cid, blockData.x);
			atomicAdd(dB + cid, blockData.y);
		}
	}
}
__global__ void GradGammaBetaSpatialKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ mean, const float* __restrict__ var, float* __restrict__ dG, float* __restrict__ dB, int N, int C, int HW){
	const int cid = blockIdx.x;
	if(cid >= C) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<PairData*>(smem);
	PairData threadData{0.0f, 0.0f};
	const int NHW = N*HW;
	const int stride = blockDim.x*gridDim.y;
	for(int nhw = tid + blockIdx.y*blockDim.x; nhw < NHW; nhw += stride){
		const int n = nhw/HW;
		const int hw = nhw % HW;
		const float varSafe = fmaxf(__ldg(var + nhw), 0.0f);
		const float invStd = rsqrtf(varSafe + EPSILON_F);
		const float m = __ldg(mean + nhw);
		const int idx = n*C*HW + cid*HW + hw;
		const float dyv = __half2float(dy[idx]);
		const float xv = __half2float(x[idx]);
		const float xnorm = (xv - m)*invStd;
		threadData.x = fmaf(xnorm, dyv, threadData.x);
		threadData.y += dyv;
	}
	threadData = WarpReducePair(threadData);
	if(laneId == 0){ warpBuffer[warpId] = threadData; }
	__syncthreads();
	if(warpId == 0){
		PairData blockData{0.0f, 0.0f};
		if(laneId < warpsPerBlock){ blockData = warpBuffer[laneId]; }
		blockData = WarpReducePair(blockData);
		if(laneId == 0){
			atomicAdd(dG + cid, blockData.x);
			atomicAdd(dB + cid, blockData.y);
		}
	}
}
__global__ void ComputeStatsKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ mean, const float* __restrict__ var, float* __restrict__ d1, float* __restrict__ d2, int N, int C, int HW){
	const int n = blockIdx.x;
	if(n >= N) return;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<PairData*>(smem);
	const float varSafe = fmaxf(var[n], 0.0f);
	const float invStd = rsqrtf(varSafe + EPSILON_F);
	const float m = mean[n];
	const int stride = C*HW;
	const int base = n*stride;
	PairData threadData{0.0f, 0.0f};
	for(int i = tid; i < stride; i += blockDim.x){
		const int c = i/HW;
		const int idx = base + i;
		const float dyv = __half2float(dy[idx]);
		const float xv = __half2float(x[idx]);
		const float gamma = __ldg(g + c);
		const float dy_g = dyv*gamma;
		const float xnorm = (xv - m)*invStd;
		threadData.x += dy_g;
		threadData.y = fmaf(dy_g, xnorm, threadData.y);
	}
	// Warp reduction
	threadData = WarpReducePair(threadData);
	if(laneId == 0){ warpBuffer[warpId] = threadData; }
	__syncthreads();
	// Final reduction
	if(warpId == 0){
		PairData blockData{0.0f, 0.0f};
		if(laneId < warpsPerBlock){ blockData = warpBuffer[laneId]; }
		blockData = WarpReducePair(blockData);
		if(laneId == 0){
			d1[n] = blockData.x;
			d2[n] = blockData.y;
		}
	}
}
__global__ void ComputeStatsSpatialKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ mean, const float* __restrict__ var, float* __restrict__ d1, float* __restrict__ d2, int N, int C, int HW){
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int laneId = tid & 31;
	const int warpsPerBlock = (blockDim.x + 31) >> 5;
	extern __shared__ unsigned char smem[];
	auto* warpBuffer = reinterpret_cast<PairData*>(smem);
	const int NHW = N*HW;
	for(int nhw = blockIdx.x; nhw < NHW; nhw += gridDim.x){
		const int n = nhw/HW;
		const int hw = nhw % HW;
		const int base = n*C*HW + hw;
		const float varSafe = fmaxf(var[nhw], 0.0f);
		const float invStd = rsqrtf(varSafe + EPSILON_F);
		const float m = mean[nhw];
		PairData threadData{0.0f, 0.0f};
		for(int c = tid; c < C; c += blockDim.x){
			const int idx = base + c*HW;
			const float dyv = __half2float(dy[idx]);
			const float xv = __half2float(x[idx]);
			const float gamma = __ldg(g + c);
			const float dy_g = dyv*gamma;
			const float xnorm = (xv - m)*invStd;
			threadData.x += dy_g;
			threadData.y = fmaf(dy_g, xnorm, threadData.y);
		}
		threadData = WarpReducePair(threadData);
		if(laneId == 0){ warpBuffer[warpId] = threadData; }
		__syncthreads();
		if(warpId == 0){
			PairData blockData{0.0f, 0.0f};
			if(laneId < warpsPerBlock){ blockData = warpBuffer[laneId]; }
			blockData = WarpReducePair(blockData);
			if(laneId == 0){
				d1[nhw] = blockData.x;
				d2[nhw] = blockData.y;
			}
		}
		__syncthreads();
	}
}
__global__ void InputGradKernel(__half* __restrict__ dx, const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ d1, const float* __restrict__ d2, const float* __restrict__ mean, const float* __restrict__ var, int N, int C, int HW){
	const int n = blockIdx.y;
	if(n >= N) return;
	const int stride = C*HW;
	const int base = n*stride;
	const float m = mean[n];
	const float varSafe = fmaxf(var[n], 0.0f);
	const float invStd = rsqrtf(varSafe + EPSILON_F);
	const float invM = 1.0f/fmaxf(static_cast<float>(stride), 1.0f);
	const float d1n = d1[n];
	const float d2n = d2[n];
	const int totalTiles = (stride + blockDim.x - 1)/blockDim.x;
	for(int tile = blockIdx.x; tile < totalTiles; tile += gridDim.x){
		const int localIdx = tile*blockDim.x + threadIdx.x;
		if(localIdx < stride){
			const int c = localIdx/HW;
			const int idx = base + localIdx;
			const float dyv = __half2float(dy[idx]);
			const float xv = __half2float(x[idx]);
			const float xnorm = (xv - m)*invStd;
			const float gi = __ldg(g + c);
			float dxv = gi*invStd*(dyv - d1n*invM - xnorm*d2n*invM);
			dxv = fmaxf(fminf(dxv, LN_GRAD_CLIP), -LN_GRAD_CLIP);
			dx[idx] = __float2half(dxv);
		}
	}
}
__global__ void InputGradSpatialKernel(__half* __restrict__ dx, const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ d1, const float* __restrict__ d2, const float* __restrict__ mean, const float* __restrict__ var, int N, int C, int HW){
	const int n = blockIdx.y;
	if(n >= N) return;
	const int stride = C*HW;
	const int base = n*stride;
	const int totalTiles = (stride + blockDim.x - 1)/blockDim.x;
	for(int tile = blockIdx.x; tile < totalTiles; tile += gridDim.x){
		const int localIdx = tile*blockDim.x + threadIdx.x;
		if(localIdx < stride){
			const int c = localIdx/HW;
			const int hw = localIdx % HW;
			const int nhw = n*HW + hw;
			const int idx = base + c*HW + hw;
			const float dyv = __half2float(dy[idx]);
			const float xv = __half2float(x[idx]);
			const float m = mean[nhw];
			const float varSafe = fmaxf(var[nhw], 0.0f);
			const float invStd = rsqrtf(varSafe + EPSILON_F);
			const float xnorm = (xv - m)*invStd;
			const float gi = __ldg(g + c);
			const float d1n = d1[nhw];
			const float d2n = d2[nhw];
			const float invM = 1.0f/fmaxf(static_cast<float>(C), 1.0f);
			float dxv = gi*invStd*(dyv - d1n*invM - xnorm*d2n*invM);
			dxv = fmaxf(fminf(dxv, LN_GRAD_CLIP), -LN_GRAD_CLIP);
			dx[idx] = __float2half(dxv);
		}
	}
}
void LayerNormBackward(__half* dx, const __half* dy, const __half* x, const float* g, float* dG, float* dB, const float* mean, const float* var, void* workspace, size_t workspaceSize, int N, int C, int HW, bool spatialMode){
	if(!dx || !dy || !x || !g || !dG || !dB || !mean || !var || !workspace){
		fprintf(stderr, "LayerNormBackward: Null pointer input\n");
		return;
	}
	if(N <= 0 || C <= 0 || HW <= 0){
		fprintf(stderr, "LayerNormBackward: Invalid dimensions N=%d, C=%d, HW=%d\n", N, C, HW);
		return;
	}
	const size_t statsCount = spatialMode ? static_cast<size_t>(N)*static_cast<size_t>(HW) : static_cast<size_t>(N);
	const size_t required_size = 2*statsCount*sizeof(float);
	if(workspaceSize < required_size){
		fprintf(stderr, "LayerNormBackward: Insufficient workspace (need %zu, got %zu)\n", required_size, workspaceSize);
		return;
	}
	auto* d1 = static_cast<float*>(workspace);
	float* d2 = d1 + statsCount;
	checkCUDA(cudaMemset(dG, 0, C*sizeof(float)));
	checkCUDA(cudaMemset(dB, 0, C*sizeof(float)));
	checkCUDA(cudaMemset(d1, 0, statsCount*sizeof(float)));
	checkCUDA(cudaMemset(d2, 0, statsCount*sizeof(float)));
	int maxSmem;
	cudaDeviceGetAttribute(&maxSmem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
	if(spatialMode){
		const int nhw = N*HW;
		const int gradTpb = SelectLayerNormThreads(max(nhw, 1));
		const int gradWarps = DivCeil(gradTpb, 32);
		const size_t gradSmemSize = gradWarps*sizeof(PairData);
		if(gradSmemSize > maxSmem){
			fprintf(stderr, "LayerNormBackward: Required shared memory %zu exceeds limit %d\n", gradSmemSize, maxSmem);
			return;
		}
		int gradGridY = min(max(nhw, 1), 65535);
		if(gradGridY < 1){ gradGridY = 1; }
		dim3 gradGrid(C, gradGridY, 1);
		GradGammaBetaSpatialKernel<<<gradGrid, gradTpb, gradSmemSize>>>(dy, x, mean, var, dG, dB, N, C, HW);
		checkCUDA(cudaGetLastError());
		const int statsTpb = SelectLayerNormThreads(C);
		const int statsWarps = DivCeil(statsTpb, 32);
		const size_t statsSmemSize = statsWarps*sizeof(PairData);
		if(statsSmemSize > maxSmem){
			fprintf(stderr, "LayerNormBackward: Required shared memory %zu exceeds limit %d\n", statsSmemSize, maxSmem);
			return;
		}
		const int statsGridX = min(max(nhw, 1), 65535);
		ComputeStatsSpatialKernel<<<statsGridX, statsTpb, statsSmemSize>>>(dy, x, g, mean, var, d1, d2, N, C, HW);
		checkCUDA(cudaGetLastError());
		const int stride = C*HW;
		const int tpb = SelectLayerNormThreads(stride);
		const int tiles = DivCeil(stride, tpb);
		const int gridX = min(max(tiles, 1), 65535);
		dim3 grid(gridX, N, 1);
		InputGradSpatialKernel<<<grid, tpb>>>(dx, dy, x, g, d1, d2, mean, var, N, C, HW);
		checkCUDA(cudaGetLastError());
	} else{
		const int gradTpb = SelectLayerNormThreads(HW);
		const int gradWarps = DivCeil(gradTpb, 32);
		const size_t gradSmemSize = gradWarps*sizeof(PairData);
		if(gradSmemSize > maxSmem){
			fprintf(stderr, "LayerNormBackward: Required shared memory %zu exceeds limit %d\n", gradSmemSize, maxSmem);
			return;
		}
		const int rowsPerBlock = max(1, min(4096/max(HW, 1), N));
		int gradGridY = min(DivCeil(N, rowsPerBlock), 65535);
		gradGridY = max(1, min(gradGridY, N));
		size_t targetBlocks = DivCeil(GS, static_cast<size_t>(gradWarps));
		if(targetBlocks == 0){ targetBlocks = 1; }
		const size_t currentBlocks = static_cast<size_t>(C)*static_cast<size_t>(gradGridY);
		if(currentBlocks < targetBlocks){
			size_t desiredGridY = DivCeil(targetBlocks, static_cast<size_t>(C));
			const size_t maxGridY = min(static_cast<size_t>(N), static_cast<size_t>(65535));
			if(desiredGridY < 1){ desiredGridY = 1; }
			if(desiredGridY > maxGridY){ desiredGridY = maxGridY; }
			gradGridY = static_cast<int>(desiredGridY);
		}
		dim3 gradGrid(C, gradGridY, 1);
		GradGammaBetaKernel<<<gradGrid, gradTpb, gradSmemSize>>>(dy, x, mean, var, dG, dB, N, C, HW);
		checkCUDA(cudaGetLastError());
		ComputeStatsKernel<<<N, gradTpb, gradSmemSize>>>(dy, x, g, mean, var, d1, d2, N, C, HW);
		checkCUDA(cudaGetLastError());
		const int stride = C*HW;
		const int tpb = SelectLayerNormThreads(stride);
		const int tiles = DivCeil(stride, tpb);
		const int gridX = min(tiles, 65535);
		dim3 grid(gridX, N, 1);
		InputGradKernel<<<grid, tpb>>>(dx, dy, x, g, d1, d2, mean, var, N, C, HW);
		checkCUDA(cudaGetLastError());
	}
}
size_t LayerNormBackwardWorkspaceSize(int N, int HW, bool spatialMode){
	const size_t statsCount = spatialMode ? static_cast<size_t>(N)*static_cast<size_t>(HW) : static_cast<size_t>(N);
	return 2*statsCount*sizeof(float);
}