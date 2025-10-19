#define __CUDACC__
#include "CuCommon.cuh"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
		data.mean += delta / data.count;
		const float delta2 = x - data.mean;
		data.m2 += delta*delta2;
		return data;
	}
	__device__ __forceinline__ WelfordData WelfordCombine(WelfordData a, const WelfordData& b){
		if(b.count == 0.0f){ return a; }
		if(a.count == 0.0f){ return b; }
		const float delta = b.mean - a.mean;
		const float count = a.count + b.count;
		a.mean += delta*(b.count / count);
		a.m2 += b.m2 + delta*delta*(a.count*b.count) / count;
		a.count = count;
		return a;
	}
	__device__ __forceinline__ WelfordData WarpReduceWelford(WelfordData value){
		for(int offset = 16; offset > 0; offset /= 2){
			WelfordData other;
			other.mean = __shfl_down_sync(0xFFFFFFFF, value.mean, offset);
			other.m2 = __shfl_down_sync(0xFFFFFFFF, value.m2, offset);
			other.count = __shfl_down_sync(0xFFFFFFFF, value.count, offset);
			value = WelfordCombine(value, other);
		}
		return value;
	}
	__device__ __forceinline__ PairData WarpReducePair(PairData value){
		for(int offset = 16; offset > 0; offset /= 2){
			value.x += __shfl_down_sync(0xFFFFFFFF, value.x, offset);
			value.y += __shfl_down_sync(0xFFFFFFFF, value.y, offset);
		}
		return value;
	}
	int SelectLayerNormThreads(int elements){
		int threads = 32;
		while(threads < elements && threads < 256){ threads <<= 1; }
		return threads;
	}
}
__global__ void ComputeMeanVarianceKernel(const __half* __restrict__ x, float* mean, float* var, int N, int C, int HW){
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
			var[n] = blockData.count > 1.0f ? blockData.m2 / blockData.count : 0.0f;
		}
	}
}
__global__ void LayerNormForwardKernel(__half* __restrict__ y, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b, const float* __restrict__ mean, const float* __restrict__ var, int N, int C, int HW){
	const int n = blockIdx.y;
	if(n >= N) return;
	const int stride = C*HW;
	const int base = n*stride;
	const float m = mean[n];
	const float invStd = rsqrtf(var[n] + EPSILON_F);
	const int totalTiles = (stride + blockDim.x - 1) / blockDim.x;
	for(int tile = blockIdx.x; tile < totalTiles; tile += gridDim.x){
		const int localIdx = tile*blockDim.x + threadIdx.x;
		if(localIdx < stride){
			const int c = localIdx / HW;
			const int idx = base + localIdx;
			const float v = __half2float(x[idx]);
			const float norm = (v - m)*invStd;
			const float gamma = __ldg(g + c);
			const float beta = __ldg(b + c);
			y[idx] = __float2half(fmaf(norm, gamma, beta));
		}
	}
}
void LayerNormForward(__half* y, const __half* x, const float* g, const float* b, float* mean, float* var, int N, int C, int HW){
	const int stride = C*HW;
	const int threads = SelectLayerNormThreads(stride);
	const int warpsPerBlock = (threads + 31) / 32;
	const size_t sm = warpsPerBlock*sizeof(WelfordData);
	ComputeMeanVarianceKernel<<<N, threads, sm>>>(x, mean, var, N, C, HW);
	auto e = cudaGetLastError();
	if(e != cudaSuccess){
		printf("LayerNorm Forward error (mean/var): %s\n", cudaGetErrorString(e));
		return;
	}
	const int tiles = (stride + threads - 1) / threads;
	int gridX = tiles;
	if(gridX > 65535){ gridX = 65535; }
	dim3 grid(gridX, N, 1);
	LayerNormForwardKernel<<<grid, threads>>>(y, x, g, b, mean, var, N, C, HW);
	e = cudaGetLastError();
	if(e != cudaSuccess){ printf("LayerNorm Forward error (norm): %s\n", cudaGetErrorString(e)); }
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
	for(int n = blockIdx.y; n < N; n += gridDim.y){
		const float invStd = rsqrtf(__ldg(var + n) + EPSILON_F);
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
	const float invStd = rsqrtf(var[n] + EPSILON_F);
	const float m = mean[n];
	const int stride = C*HW;
	const int base = n*stride;
	PairData threadData{0.0f, 0.0f};
	for(int i = tid; i < stride; i += blockDim.x){
		const int c = i / HW;
		const int idx = base + i;
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
			d1[n] = blockData.x;
			d2[n] = blockData.y;
		}
	}
}
#define LN_GRAD_CLIP 5.0f
__global__ void InputGradKernel(__half* __restrict__ dx, const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ d1, const float* __restrict__ d2, const float* __restrict__ mean, const float* __restrict__ var, int N, int C, int HW){
	const int n = blockIdx.y;
	if(n >= N) return;
	const int stride = C*HW;
	const int base = n*stride;
	const float m = mean[n];
	const float invStd = rsqrtf(var[n] + EPSILON_F);
	const float invM = 1.0f / static_cast<float>(stride);
	const float d1n = d1[n];
	const float d2n = d2[n];
	const int totalTiles = (stride + blockDim.x - 1) / blockDim.x;
	for(int tile = blockIdx.x; tile < totalTiles; tile += gridDim.x){
		const int localIdx = tile*blockDim.x + threadIdx.x;
		if(localIdx < stride){
			const int c = localIdx / HW;
			const int idx = base + localIdx;
			const float dyv = __half2float(dy[idx]);
			const float xv = __half2float(x[idx]);
			const float xnorm = (xv - m)*invStd;
			const float gi = __ldg(g + c);
			const float dxv = gi*invStd*(dyv - d1n*invM - xnorm*d2n*invM);
			const float clipped = fmaxf(fminf(dxv, LN_GRAD_CLIP), -LN_GRAD_CLIP);
			dx[idx] = __float2half(clipped);
		}
	}
}
void LayerNormBackward(__half* dx, const __half* dy, const __half* x, const float* g, float* dG, float* dB, const float* mean, const float* var, void* workspace, size_t workspace_size, int N, int C, int HW){
	const auto required_size = 2*N*sizeof(float);
	if(workspace_size < required_size){
		printf("LayerNorm Backward error: insufficient workspace (need %zu, got %zu)\n", required_size, workspace_size);
		return;
	}
	const auto d1 = static_cast<float*>(workspace);
	float* d2 = d1 + N;
	cudaMemset(dG, 0, C*sizeof(float));
	cudaMemset(dB, 0, C*sizeof(float));
	cudaMemset(d1, 0, N*sizeof(float));
	cudaMemset(d2, 0, N*sizeof(float));
	const int gradThreads = SelectLayerNormThreads(HW);
	const int gradWarps = (gradThreads + 31) / 32;
	const size_t gradSm = gradWarps*sizeof(PairData);
	int rowsPerBlockTarget = (HW > 0) ? (4096 / HW) : 4096;
	if(rowsPerBlockTarget < 1){ rowsPerBlockTarget = 1; }
	int gradGridY = (N + rowsPerBlockTarget - 1) / rowsPerBlockTarget;
	if(gradGridY > N){ gradGridY = N; }
	if(gradGridY < 1){ gradGridY = 1; }
	if(gradGridY > 65535){ gradGridY = 65535; }
	dim3 gradGrid(C, gradGridY, 1);
	GradGammaBetaKernel<<<gradGrid, gradThreads, gradSm>>>(dy, x, mean, var, dG, dB, N, C, HW);
	auto e = cudaGetLastError();
	if(e != cudaSuccess){
		printf("LayerNorm Backward error (gamma/beta): %s\n", cudaGetErrorString(e));
		return;
	}
	ComputeStatsKernel<<<N, gradThreads, gradSm>>>(dy, x, g, mean, var, d1, d2, N, C, HW);
	e = cudaGetLastError();
	if(e != cudaSuccess){
		printf("LayerNorm Backward error (stats): %s\n", cudaGetErrorString(e));
		return;
	}
	const int stride = C*HW;
	const int threads = SelectLayerNormThreads(stride);
	const int tiles = (stride + threads - 1) / threads;
	int gridX = tiles;
	if(gridX > 65535){ gridX = 65535; }
	dim3 grid(gridX, N, 1);
	InputGradKernel<<<grid, threads>>>(dx, dy, x, g, d1, d2, mean, var, N, C, HW);
	e = cudaGetLastError();
	if(e != cudaSuccess){ printf("LayerNorm Backward error (input): %s\n", cudaGetErrorString(e)); }
}