#include "CuCommon.cuh"
__global__ void ComputeMeanVarianceKernel(const __half* __restrict__ x, float* mean, float* var, int N, int C, int HW){
	extern __shared__ float sm[];
	float* sMean = sm, *sM2 = sMean+blockDim.x, *sCnt = sM2+blockDim.x;
	const int tid = threadIdx.x, n = blockIdx.x; if(n>=N) return;
	float m = 0.f, m2 = 0.f, c = 0.f;
	const int stride = C*HW;
	for(int i = tid; i<stride; i += blockDim.x){
		const float v = __half2float(x[n*stride+i]); ++c;
		const float d = v-m; m += d/c; m2 += d*(v-m);
	}
	sMean[tid] = m; sM2[tid] = m2; sCnt[tid] = c; __syncthreads();
	for(int s = blockDim.x>>1; s; s >>= 1){
		if(tid<s){
			const float m1 = sMean[tid], m2_ = sMean[tid+s], c1 = sCnt[tid], c2 = sCnt[tid+s];
			const float d = m2_-m1, nc = c1+c2;
			sMean[tid] = (m1*c1+m2_*c2)/nc;
			sM2[tid] = sM2[tid]+sM2[tid+s]+d*d*c1*c2/nc;
			sCnt[tid] = nc;
		}
		__syncthreads();
	}
	if(!tid){ mean[n] = sMean[0]; var[n] = sM2[0]/sCnt[0]; }
}
__global__ void LayerNormForwardKernel(__half* __restrict__ y, const __half* __restrict__ x, const float* g, const float* b, const float* mean, const float* var, int N, int C, int HW){
	extern __shared__ float sp[];
	float* sg = sp, *sb = sg+C;
	for(int i = threadIdx.x; i<C; i += blockDim.x){ sg[i] = g[i]; sb[i] = b[i]; }
	__syncthreads();
	const int tot = N*C*HW;
	for(int idx = blockIdx.x*blockDim.x+threadIdx.x; idx<tot; idx += blockDim.x*gridDim.x){
		const int c = (idx/HW)%C, n = idx/(C*HW);
		const float v = __half2float(x[idx]);
		const float nrm = (v-mean[n])*rsqrtf(var[n]+EPSILON_F);
		y[idx] = __float2half(nrm*sg[c]+sb[c]);
	}
}
void LayerNormForward(__half* y, const __half* x, const float* g, const float* b, float* mean, float* var, int N, int C, int HW){
	int sm = 3*BS*sizeof(float);
	ComputeMeanVarianceKernel<<<N, BS, sm>>>(x, mean, var, N, C, HW);
	int grids = DivCeil(N*C*HW, BS);
	sm = 2*C*sizeof(float);
	LayerNormForwardKernel<<<grids, BS, sm>>>(y, x, g, b, mean, var, N, C, HW);
	const cudaError_t e = cudaGetLastError(); if(e) printf("LN Fwd err %s\n", cudaGetErrorString(e));
}
__global__ void GradGammaBetaDotKernel(const __half* __restrict__ dy, const __half* __restrict__ x, const float* g, float* dG, float* dB, float* d1, float* d2, const float* mean, const float* var, int N, int C, int HW){
	extern __shared__ float sm[];
	float* sG = sm, *sB = sG+blockDim.x;
	const int cid = blockIdx.x, tid = threadIdx.x; if(cid>=C) return;
	float tG = 0.f, tB = 0.f;
	for(int n = 0; n<N; ++n){
		const float inv = rsqrtf(var[n]+EPSILON_F);
		for(int i = tid; i<HW; i += blockDim.x){
			const int idx = n*C*HW+cid*HW+i;
			const float dyv = __half2float(dy[idx]);
			const float xh = (__half2float(x[idx])-mean[n])*inv;
			const float dy_g = dyv*g[cid];
			atomicAdd(&d1[n], dy_g);
			atomicAdd(&d2[n], dy_g*xh);
			tG += xh*dyv;
			tB += dyv;
		}
	}
	sG[tid] = tG; sB[tid] = tB; __syncthreads();
	for(int s = blockDim.x>>1; s; s >>= 1){
		if(tid<s){ sG[tid] += sG[tid+s]; sB[tid] += sB[tid+s]; }
		__syncthreads();
	}
	if(!tid){ dG[cid] = sG[0]; dB[cid] = sB[0]; }
}
__global__ void InputGradKernel(__half* __restrict__ dx, const __half* __restrict__ x, const float* g, const float* d1, const float* d2, const float* mean, const float* var, int N, int C, int HW){
	const int tot = N*C*HW;
	const float invM = 1.f/(C*HW);
	for(int idx = blockIdx.x*blockDim.x+threadIdx.x; idx<tot; idx += blockDim.x*gridDim.x){
		const int c = (idx/HW)%C, n = idx/(C*HW);
		const float inv = rsqrtf(var[n]+EPSILON_F);
		const float v = __half2float(x[idx]);
		const float dyv = __half2float(dx[idx]);
		const float xh = (v-mean[n])*inv;
		const float gi = g[c];
		const float dxv = gi*inv*(dyv-d1[n]*invM-xh*d2[n]*invM);
		dx[idx] = __float2half(dxv);
	}
}
void LayerNormBackward(__half* dx, const __half* x, const float* g, float* dG, float* dB, const float* mean, const float* var, int N, int C, int HW){
	float* d1, *d2;
	cudaMalloc(&d1, N*sizeof(float)); cudaMalloc(&d2, N*sizeof(float));
	cudaMemset(d1, 0, N*sizeof(float)); cudaMemset(d2, 0, N*sizeof(float));
	int sm = 2*BS*sizeof(float);
	cudaMemset(dG, 0, C*sizeof(float)); cudaMemset(dB, 0, C*sizeof(float));
	GradGammaBetaDotKernel<<<C, BS, sm>>>(dx, x, g, dG, dB, d1, d2, mean, var, N, C, HW);
	int grids = DivCeil(N*C*HW, BS);
	InputGradKernel<<<grids, BS>>>(dx, x, g, d1, d2, mean, var, N, C, HW);
	cudaFree(d1); cudaFree(d2);
	const cudaError_t e = cudaGetLastError(); if(e) printf("LN Bwd err %s\n", cudaGetErrorString(e));
}