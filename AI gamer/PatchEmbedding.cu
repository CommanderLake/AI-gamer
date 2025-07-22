#include "CuCommon.cuh"
#include <cstdio>
#include <cuda_runtime_api.h>
__global__ void ExtractPatchesKernelVec2(const __half* __restrict__ x, __half* __restrict__ y, int B, int C, int H, int W, int P){
	const int kPatchArea = P*P;
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total_elements = static_cast<long>(B)*PH*PW*patchDim;
	const long total_vec2 = total_elements/2;
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= total_vec2) return;
	// Process 2 elements at once
	__half2 vals = __float2half2_rn(0.0f);
#pragma unroll 2
	for(int i = 0; i < 2; i++){
		long elem_idx = idx*2 + i;
		if(elem_idx >= total_elements) break;
		// Same decomposition logic as above
		long patch = elem_idx/patchDim;
		int inPatch = elem_idx - patch*patchDim;
		int b = patch/(PH*PW);
		int pp = patch - b*(PH*PW);
		int pyPatch = pp/PW;
		int pxPatch = pp - pyPatch*PW;
		int c = inPatch/kPatchArea;
		int rem = inPatch - c*kPatchArea;
		int py = rem/P;
		int px = rem - py*P;
		int srcY = pyPatch*P + py;
		int srcX = pxPatch*P + px;
		if(srcY < H && srcX < W){
			long srcIdx = (((static_cast<long>(b)*C + c)*H + srcY)*W) + srcX;
			if(i == 0) vals.x = x[srcIdx];
			else vals.y = x[srcIdx];
		}
	}
	reinterpret_cast<__half2*>(y)[idx] = vals;
}
__global__ void ExtractPatchesKernel(const __half* __restrict__ x, __half* __restrict__ y, int B, int C, int H, int W, int P){
	const int kPatchArea = P*P;
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total = static_cast<long>(B)*PH*PW*patchDim;
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= total) return;
	// Decompose the output index
	long patch = idx/patchDim;
	int inPatch = idx - patch*patchDim;
	int b = patch/(PH*PW);
	int pp = patch - b*(PH*PW);
	int pyPatch = pp/PW;
	int pxPatch = pp - pyPatch*PW;
	int c = inPatch/kPatchArea;
	int rem = inPatch - c*kPatchArea;
	int py = rem/P;
	int px = rem - py*P;
	int srcY = pyPatch*P + py;
	int srcX = pxPatch*P + px;
	__half val = __float2half(0.0f);
	if(srcY < H && srcX < W){
		long srcIdx = (((static_cast<long>(b)*C + c)*H + srcY)*W) + srcX;
		val = x[srcIdx];
	}
	y[idx] = val;
}
void ExtractPatches(const __half* in, __half* out, int B, int C, int H, int W, int P){
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const size_t total = static_cast<size_t>(B)*PH*PW*C*P*P;
	if(total % 2 == 0){
		int blocks = 0, tpb = 0;
		GetLaunchConfig(total/2, blocks, tpb);
		ExtractPatchesKernelVec2<<<blocks, tpb>>>(in, out, B, C, H, W, P);
	} else{
		int blocks = 0, tpb = 0;
		GetLaunchConfig(total, blocks, tpb);
		ExtractPatchesKernel<<<blocks, tpb>>>(in, out, B, C, H, W, P);
	}
	cudaDeviceSynchronize();
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("ExtractPatches error: %s\n", cudaGetErrorString(e));
}
__global__ void CombinePatchGradsKernel(const __half* __restrict__ dy, __half* __restrict__ dx, int B, int C, int H, int W, int P){
	const int kPatchArea = P*P;
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total = static_cast<long>(B)*PH*PW*patchDim;
	long idx = blockIdx.x*blockDim.x + threadIdx.x;
	if(idx >= total) return;
	long patch = idx/patchDim;
	int inPatch = idx - patch*patchDim;
	int b = patch/(PH*PW);
	int pp = patch - b*(PH*PW);
	int pyPatch = pp/PW;
	int pxPatch = pp - pyPatch*PW;
	int c = inPatch/kPatchArea;
	int rem = inPatch - c*kPatchArea;
	int py = rem/P;
	int px = rem - py*P;
	int dstY = pyPatch*P + py;
	int dstX = pxPatch*P + px;
	if(dstY >= H || dstX >= W) return;
	long dstIdx = (((static_cast<long>(b)*C + c)*H + dstY)*W) + dstX;
	// For non-overlapping patches, direct assignment is correct
	// If you need overlapping patches, use atomicAdd
	dx[dstIdx] = dy[idx];
}
void CombinePatchGrads(const __half* dy, __half* dx, int B, int C, int H, int W, int P){
	cudaMemset(dx, 0, B*C*H*W*sizeof(__half));
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const size_t total = static_cast<size_t>(B)*PH*PW*C*P*P;
	int blocks = 0, tpb = 0;
	GetLaunchConfig(total, blocks, tpb);
	CombinePatchGradsKernel<<<blocks, tpb>>>(dy, dx, B, C, H, W, P);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("CombinePatchGrads error: %s\n", cudaGetErrorString(e));
}
__global__ void SumPositionalGradKernel(const __half* grad, __half* out, int B, int C, int P, bool first){
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	const int total = C*P;
	if(idx >= total) return;
	const int c = idx / P;
	const int p = idx - c*P;
	float sum = 0.0f;
	for(int b = 0; b < B; ++b){
		const int index = c + C*(b*P + p);
		sum += __half2float(grad[index]);
	}
	if(first) out[idx] = __float2half(sum); else out[idx] = __float2half(__half2float(out[idx]) + sum);
}

void SumPositionalGrad(const __half* grad, __half* out, int B, int C, int P, bool first){
	int blocks = 0, tpb = 0;
	GetLaunchConfig(C*P, blocks, tpb);
	SumPositionalGradKernel<<<blocks, tpb>>>(grad, out, B, C, P, first);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("SumPositionalGrad error: %s\n", cudaGetErrorString(e));
}