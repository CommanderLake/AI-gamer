#include "CuCommon.cuh"
__device__ __forceinline__ float warpReduceSum(float val){
	for(int offset = 16; offset > 0; offset /= 2){ val += __shfl_down_sync(0xFFFFFFFF, val, offset); }
	return val;
}
__global__ void ComputeMeanVarianceKernel(const __half* __restrict__ x, float* mean, float* var, int N, int C, int HW){
	extern __shared__ float sm[];
	const int tid = threadIdx.x;
	const int n = blockIdx.x;
	if(n >= N) return;
	const int warpId = tid / 32;
	const int laneId = tid % 32;
	const int warpsPerBlock = (blockDim.x + 31) / 32;
	float* sMean = sm;
	float* sM2 = sMean + warpsPerBlock;
	float* sCnt = sM2 + warpsPerBlock;
	float m = 0.0f, m2 = 0.0f, c = 0.0f;
	const int stride = C*HW;
	const __half* xn = x + n*stride;
	for(int i = tid; i < stride; i += blockDim.x){
		const float v = __half2float(xn[i]);
		c += 1.0f;
		const float d = v - m;
		m += d / c;
		m2 += d*(v - m);
	}
	__syncwarp();
	float total_c = warpReduceSum(c);
	float weighted_mean = 0.0f;
	if(total_c > 0){ weighted_mean = warpReduceSum(m*c) / total_c; }
	float combined_m2 = 0.0f;
	for(int offset = 16; offset > 0; offset /= 2){
		float other_m = __shfl_down_sync(0xFFFFFFFF, m, offset);
		float other_m2 = __shfl_down_sync(0xFFFFFFFF, m2, offset);
		float other_c = __shfl_down_sync(0xFFFFFFFF, c, offset);
		if(laneId + offset < 32 && other_c > 0){
			float delta = other_m - m;
			float new_c = c + other_c;
			if(new_c > 0){
				combined_m2 = m2 + other_m2 + delta*delta*c*other_c / new_c;
				m2 = combined_m2;
				c = new_c;
			}
		}
	}
	if(laneId == 0){
		sMean[warpId] = weighted_mean;
		sM2[warpId] = (total_c > 0) ? m2 : 0.0f;
		sCnt[warpId] = total_c;
	}
	__syncthreads();
	if(tid < warpsPerBlock){
		float warp_mean = sMean[tid];
		float warp_m2 = sM2[tid];
		float warp_c = sCnt[tid];
		float total_count = warpReduceSum(warp_c);
		float final_mean = 0.0f;
		if(total_count > 0){ final_mean = warpReduceSum(warp_mean*warp_c) / total_count; }
		float final_m2 = 0.0f;
		for(int offset = 16; offset > 0; offset /= 2){
			float other_mean = __shfl_down_sync(0xFFFFFFFF, warp_mean, offset);
			float other_m2 = __shfl_down_sync(0xFFFFFFFF, warp_m2, offset);
			float other_c = __shfl_down_sync(0xFFFFFFFF, warp_c, offset);
			if(tid + offset < warpsPerBlock && other_c > 0){
				float delta = other_mean - warp_mean;
				float new_c = warp_c + other_c;
				if(new_c > 0){
					final_m2 = warp_m2 + other_m2 + delta*delta*warp_c*other_c / new_c;
					warp_m2 = final_m2;
					warp_c = new_c;
				}
			}
		}
		if(tid == 0){
			mean[n] = (total_count > 0) ? final_mean : 0.0f;
			var[n] = (total_count > 1) ? final_m2 / total_count : 0.0f;
		}
	}
}
__global__ void LayerNormForwardKernel(__half* __restrict__ y, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ b, const float* __restrict__ mean, const float* __restrict__ var, int N, int C, int HW){
	extern __shared__ float sp[];
	float* sg = sp;
	float* sb = sg + C;
	for(int i = threadIdx.x; i < C; i += blockDim.x){
		sg[i] = g[i];
		sb[i] = b[i];
	}
	__syncthreads();
	const int tot = N*C*HW;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < tot; idx += blockDim.x*gridDim.x){
		const int hw = idx % HW;
		const int c = (idx / HW) % C;
		const int n = idx / (C*HW);
		const float v = __half2float(x[idx]);
		const float m = mean[n];
		const float invStd = rsqrtf(var[n] + EPSILON_F);
		const float norm = (v - m)*invStd;
		y[idx] = __float2half(fmaf(norm, sg[c], sb[c]));
	}
}
void LayerNormForward(__half* y, const __half* x, const float* g, const float* b, float* mean, float* var, int N, int C, int HW){
	int tpb = BS;
	const int warpsPerBlock = (tpb + 31) / 32;
	int sm = 3*warpsPerBlock*sizeof(float);
	ComputeMeanVarianceKernel<<<N, tpb, sm>>>(x, mean, var, N, C, HW);
	auto e = cudaGetLastError();
	if(e != cudaSuccess){
		printf("LayerNorm Forward error (mean/var): %s\n", cudaGetErrorString(e));
		return;
	}
	int grids = 0;
	GetLaunchConfig(N*C*HW, grids, tpb);
	sm = 2*C*sizeof(float);
	LayerNormForwardKernel<<<grids, tpb, sm>>>(y, x, g, b, mean, var, N, C, HW);
	e = cudaGetLastError();
	if(e != cudaSuccess){ printf("LayerNorm Forward error (norm): %s\n", cudaGetErrorString(e)); }
}
__global__ void GradGammaBetaKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ mean, const float* __restrict__ var, float* __restrict__ dG, float* __restrict__ dB, int N, int C, int HW){
	extern __shared__ float sm[];
	const int cid = blockIdx.x;
	const int tid = threadIdx.x;
	if(cid >= C) return;
	const int warpId = tid / 32;
	const int laneId = tid % 32;
	const int warpsPerBlock = (blockDim.x + 31) / 32;
	float* sG = sm;
	float* sB = sG + warpsPerBlock;
	float tG = 0.0f, tB = 0.0f;
	for(int n = 0; n < N; ++n){
		const float invStd = rsqrtf(var[n] + EPSILON_F);
		const float m = mean[n];
		const int base = n*C*HW + cid*HW;
		for(int i = tid; i < HW; i += blockDim.x){
			const int idx = base + i;
			const float dyv = __half2float(dy[idx]);
			const float xv = __half2float(x[idx]);
			const float xnorm = (xv - m)*invStd;
			tG = fmaf(xnorm, dyv, tG);
			tB += dyv;
		}
	}
	__syncwarp();
	tG = warpReduceSum(tG);
	tB = warpReduceSum(tB);
	if(laneId == 0){
		sG[warpId] = tG;
		sB[warpId] = tB;
	}
	__syncthreads();
	if(tid < warpsPerBlock){
		float finalG = sG[tid];
		float finalB = sB[tid];
		__syncwarp();
		finalG = warpReduceSum(finalG);
		finalB = warpReduceSum(finalB);
		if(tid == 0){
			dG[cid] = finalG;
			dB[cid] = finalB;
		}
	}
}
__global__ void ComputeStatsKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ mean, const float* __restrict__ var, float* __restrict__ d1, float* __restrict__ d2, int N, int C, int HW){
	extern __shared__ float sm[];
	const int n = blockIdx.x;
	const int tid = threadIdx.x;
	if(n >= N) return;
	const int warpId = tid / 32;
	const int laneId = tid % 32;
	const int warpsPerBlock = (blockDim.x + 31) / 32;
	float* sD1 = sm;
	float* sD2 = sD1 + warpsPerBlock;
	float tD1 = 0.0f, tD2 = 0.0f;
	const float invStd = rsqrtf(var[n] + EPSILON_F);
	const float m = mean[n];
	const int stride = C*HW;
	const int base = n*stride;
	for(int i = tid; i < stride; i += blockDim.x){
		const int c = (i / HW) % C;
		const int idx = base + i;
		const float dyv = __half2float(dy[idx]);
		const float xv = __half2float(x[idx]);
		const float xnorm = (xv - m)*invStd;
		const float dy_g = dyv*g[c];
		tD1 += dy_g;
		tD2 = fmaf(dy_g, xnorm, tD2);
	}
	__syncwarp();
	tD1 = warpReduceSum(tD1);
	tD2 = warpReduceSum(tD2);
	if(laneId == 0){
		sD1[warpId] = tD1;
		sD2[warpId] = tD2;
	}
	__syncthreads();
	if(tid < warpsPerBlock){
		float finalD1 = sD1[tid];
		float finalD2 = sD2[tid];
		__syncwarp();
		finalD1 = warpReduceSum(finalD1);
		finalD2 = warpReduceSum(finalD2);
		if(tid == 0){
			d1[n] = finalD1;
			d2[n] = finalD2;
		}
	}
}
#define LN_GRAD_CLIP 5.0f
__global__ void InputGradKernel(__half* __restrict__ dx, const __half* __restrict__ dy, const __half* __restrict__ x, const float* __restrict__ g, const float* __restrict__ d1, const float* __restrict__ d2, const float* __restrict__ mean, const float* __restrict__ var, int N, int C, int HW){
	const int tot = N*C*HW;
	const float invM = 1.0f / (C*HW);
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < tot; idx += blockDim.x*gridDim.x){
		const int hw = idx % HW;
		const int c = (idx / HW) % C;
		const int n = idx / (C*HW);
		const float invStd = rsqrtf(var[n] + EPSILON_F);
		const float m = mean[n];
		const float xv = __half2float(x[idx]);
		const float dyv = __half2float(dy[idx]);
		const float xnorm = (xv - m)*invStd;
		const float gi = g[c];
		const float dxv = gi*invStd*(dyv - d1[n]*invM - xnorm*d2[n]*invM);
		const float clipped = fmaxf(fminf(dxv, LN_GRAD_CLIP), -LN_GRAD_CLIP);
		dx[idx] = __float2half(clipped);
	}
}
void LayerNormBackward(__half* dx, const __half* dy, const __half* x, const float* g, float* dG, float* dB, const float* mean, const float* var, void* workspace, size_t workspace_size, int N, int C, int HW){
	const auto required_size = 2*N*sizeof(float);
	if(workspace_size < required_size){
		printf("LayerNorm Backward error: insufficient workspace (need %zu, got %zu)\n", required_size, workspace_size);
		return;
	}
	auto d1 = static_cast<float*>(workspace);
	float* d2 = d1 + N;
	// Clear gradients
	cudaMemset(dG, 0, C*sizeof(float));
	cudaMemset(dB, 0, C*sizeof(float));
	cudaMemset(d1, 0, N*sizeof(float));
	cudaMemset(d2, 0, N*sizeof(float));
	int tpb = BS;
	const int warpsPerBlock = (tpb + 31) / 32;
	int sm = 2*warpsPerBlock*sizeof(float);
	GradGammaBetaKernel<<<C, tpb, sm>>>(dy, x, mean, var, dG, dB, N, C, HW);
	auto e = cudaGetLastError();
	if(e != cudaSuccess){
		printf("LayerNorm Backward error (gamma/beta): %s\n", cudaGetErrorString(e));
		return;
	}
	sm = 2*warpsPerBlock*sizeof(float);
	ComputeStatsKernel<<<N, tpb, sm>>>(dy, x, g, mean, var, d1, d2, N, C, HW);
	e = cudaGetLastError();
	if(e != cudaSuccess){
		printf("LayerNorm Backward error (stats): %s\n", cudaGetErrorString(e));
		return;
	}
	int grids = 0;
	GetLaunchConfig(N*C*HW, grids, tpb);
	InputGradKernel<<<grids, tpb>>>(dx, dy, x, g, d1, d2, mean, var, N, C, HW);
	e = cudaGetLastError();
	if(e != cudaSuccess){ printf("LayerNorm Backward error (input): %s\n", cudaGetErrorString(e)); }
}