#include "CuCommon.h"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
__global__ void ExtractPatchesKernelVec2(const __half* __restrict__ x, __half* __restrict__ y, int B, int C, int H, int W, int P){
	const int kPatchArea = P*P;
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total_elements = static_cast<long>(B)*PH*PW*patchDim;
	const long total_vec2 = total_elements/2;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	for(long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total_vec2; idx += stride){
		__half2 vals = __float2half2_rn(0.0f);
#pragma unroll 2
		for(int i = 0; i < 2; i++){
			const long elem_idx = idx*2 + i;
			if(elem_idx >= total_elements) break;
			const long patch = elem_idx/patchDim;
			const int inPatch = elem_idx - patch*patchDim;
			const int b = patch/(PH*PW);
			const int pp = patch - b*(PH*PW);
			const int pyPatch = pp/PW;
			const int pxPatch = pp - pyPatch*PW;
			const int c = inPatch/kPatchArea;
			const int rem = inPatch - c*kPatchArea;
			const int py = rem/P;
			const int px = rem - py*P;
			const int srcY = pyPatch*P + py;
			const int srcX = pxPatch*P + px;
			if(srcY < H && srcX < W){
				const long srcIdx = (((static_cast<long>(b)*C + c)*H + srcY)*W) + srcX;
				if(i == 0) vals.x = x[srcIdx];
				else vals.y = x[srcIdx];
			}
		}
		reinterpret_cast<__half2*>(y)[idx] = vals;
	}
}
__global__ void ExtractPatchesKernel(const __half* __restrict__ x, __half* __restrict__ y, int B, int C, int H, int W, int P){
	const int kPatchArea = P*P;
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total = static_cast<long>(B)*PH*PW*patchDim;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	for(long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const long patch = idx/patchDim;
		const int inPatch = idx - patch*patchDim;
		const int b = patch/(PH*PW);
		const int pp = patch - b*(PH*PW);
		const int pyPatch = pp/PW;
		const int pxPatch = pp - pyPatch*PW;
		const int c = inPatch/kPatchArea;
		const int rem = inPatch - c*kPatchArea;
		const int py = rem/P;
		const int px = rem - py*P;
		const int srcY = pyPatch*P + py;
		const int srcX = pxPatch*P + px;
		__half val = __float2half(0.0f);
		if(srcY < H && srcX < W){
			const long srcIdx = (((static_cast<long>(b)*C + c)*H + srcY)*W) + srcX;
			val = x[srcIdx];
		}
		y[idx] = val;
	}
}
void ExtractPatches(const __half* in, __half* out, int B, int C, int H, int W, int P){
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const size_t total = static_cast<size_t>(B)*PH*PW*C*P*P;
	if(total == 0) return;
	if(total % 2 == 0){
		size_t blocks = 0, tpb = 0;
		GetLaunchConfigGridStride(total/2, blocks, tpb);
		if(blocks > 0 && tpb > 0) ExtractPatchesKernelVec2<<<blocks, tpb>>>(in, out, B, C, H, W, P);
	} else{
		size_t blocks = 0, tpb = 0;
		GetLaunchConfigGridStride(total, blocks, tpb);
		if(blocks > 0 && tpb > 0) ExtractPatchesKernel<<<blocks, tpb>>>(in, out, B, C, H, W, P);
	}
	checkCUDA(cudaGetLastError());
}
__global__ void CombinePatchGradsKernel(const __half* __restrict__ dy, __half* __restrict__ dx, int B, int C, int H, int W, int P){
	const int kPatchArea = P*P;
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total = static_cast<long>(B)*PH*PW*patchDim;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	for(long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const long patch = idx/patchDim;
		const int inPatch = idx - patch*patchDim;
		const int b = patch/(PH*PW);
		const int pp = patch - b*(PH*PW);
		const int pyPatch = pp/PW;
		const int pxPatch = pp - pyPatch*PW;
		const int c = inPatch/kPatchArea;
		const int rem = inPatch - c*kPatchArea;
		const int py = rem/P;
		const int px = rem - py*P;
		const int dstY = pyPatch*P + py;
		const int dstX = pxPatch*P + px;
		if(dstY >= H || dstX >= W) continue;
		const long dstIdx = (((static_cast<long>(b)*C + c)*H + dstY)*W) + dstX;
		dx[dstIdx] = dy[idx];
	}
}
void CombinePatchGrads(const __half* dy, __half* dx, int B, int C, int H, int W, int P){
	checkCUDA(cudaMemset(dx, 0, B*C*H*W*sizeof(__half)));
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const size_t total = static_cast<size_t>(B)*PH*PW*C*P*P;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	CombinePatchGradsKernel<<<blocks, tpb>>>(dy, dx, B, C, H, W, P);
	checkCUDA(cudaGetLastError());
}
__global__ void SumPositionalGradKernel(const __half* grad, __half* out, int B, int C, int P, bool first, float scale){
	const int total = C*P;
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int c = idx/P;
		const int p = idx - c*P;
		float sum = 0.0f;
		for(int b = 0; b < B; ++b){
			const int index = c + C*(b*P + p);
			sum += __half2float(grad[index]);
		}
		const float scaled = sum*scale;
		if(first){
			out[idx] = __float2half(scaled);
		} else{
			const float prev = __half2float(out[idx]);
			out[idx] = __float2half(prev + scaled);
		}
	}
}
void SumPositionalGrad(const __half* grad, __half* out, int B, int C, int P, bool first, float scale){
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(C*P, blocks, tpb);
	SumPositionalGradKernel<<<blocks, tpb>>>(grad, out, B, C, P, first, scale);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("SumPositionalGrad error: %s\n", cudaGetErrorString(e));
}
__global__ void AddTemporalPositionalEmbeddingKernel(__half* data, const __half* temporalPos, int batchSize, int temporalLength, int numPatches, int embedDim){
	const long total = static_cast<long>(batchSize)*temporalLength*numPatches*embedDim;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	const int perTime = numPatches*embedDim;
	for(long idx = static_cast<long>(blockIdx.x)*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int featureIndex = static_cast<int>(idx % embedDim);
		const int sampleIndex = static_cast<int>(idx / perTime);
		const int t = sampleIndex % temporalLength;
		const int temporalOffset = t*embedDim + featureIndex;
		data[idx] = __hadd(data[idx], temporalPos[temporalOffset]);
	}
}
void AddTemporalPositionalEmbedding(__half* data, const __half* temporalPos, int batchSize, int temporalLength, int numPatches, int embedDim){
	if(data == nullptr || temporalPos == nullptr || batchSize <= 0 || temporalLength <= 1 || numPatches <= 0 || embedDim <= 0) return;
	const size_t total = static_cast<size_t>(batchSize)*temporalLength*numPatches*embedDim;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	if(blocks > 0 && tpb > 0){ AddTemporalPositionalEmbeddingKernel<<<blocks, tpb>>>(data, temporalPos, batchSize, temporalLength, numPatches, embedDim); }
	checkCUDA(cudaGetLastError());
}
__global__ void SumTemporalPositionalGradKernel(const __half* grad, __half* out, int batchSize, int temporalLength, int numPatches, int embedDim, bool first, float scale){
	const int total = temporalLength*embedDim;
	const int stride = blockDim.x*gridDim.x;
	const int tokensPerTime = numPatches*embedDim;
	const int tokensPerBatch = temporalLength*tokensPerTime;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int t = idx/embedDim;
		const int c = idx - t*embedDim;
		float sum = 0.0f;
		for(int b = 0; b < batchSize; ++b){
			const int base = b*tokensPerBatch + t*tokensPerTime;
			for(int p = 0; p < numPatches; ++p){
				sum += __half2float(grad[base + p*embedDim + c]);
			}
		}
		const float scaled = sum*scale;
		if(first){ out[idx] = __float2half(scaled); }
		else{ out[idx] = __float2half(__half2float(out[idx]) + scaled); }
	}
}
void SumTemporalPositionalGrad(const __half* grad, __half* out, int batchSize, int temporalLength, int numPatches, int embedDim, bool first, float scale){
	if(grad == nullptr || out == nullptr || batchSize <= 0 || temporalLength <= 1 || numPatches <= 0 || embedDim <= 0) return;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(static_cast<size_t>(temporalLength)*embedDim, blocks, tpb);
	if(blocks > 0 && tpb > 0){ SumTemporalPositionalGradKernel<<<blocks, tpb>>>(grad, out, batchSize, temporalLength, numPatches, embedDim, first, scale); }
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("SumTemporalPositionalGrad error: %s\n", cudaGetErrorString(e));
}
