#include "CuCommon.cuh"
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
			long elem_idx = idx*2 + i;
			if(elem_idx >= total_elements) break;
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
}
__global__ void ExtractPatchesKernel(const __half* __restrict__ x, __half* __restrict__ y, int B, int C, int H, int W, int P){
	const int kPatchArea = P*P;
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total = static_cast<long>(B)*PH*PW*patchDim;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	for(long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
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
		if(dstY >= H || dstX >= W) continue;
		long dstIdx = (((static_cast<long>(b)*C + c)*H + dstY)*W) + dstX;
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
__global__ void AddPerTokenEmbeddingKernel(__half* output, const __half* embed, int batch, int tokens, int embedDim){
	const size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	if(idx >= total) return;
	const int feature = idx % embedDim;
	const size_t tokenIndex = idx/embedDim;
	const int token = static_cast<int>(tokenIndex % tokens);
	const float sum = __half2float(output[idx]) + __half2float(embed[token*embedDim + feature]);
	output[idx] = __float2half(sum);
}
void AddPerTokenEmbedding(__half* output, const __half* embed, int batch, int tokens, int embedDim){
	const size_t total = static_cast<size_t>(batch)*tokens*embedDim;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	if(blocks == 0 || tpb == 0) return;
	AddPerTokenEmbeddingKernel<<<blocks, tpb>>>(output, embed, batch, tokens, embedDim);
	checkCUDA(cudaGetLastError());
}

__global__ void TemporalFusePatchesKernel(const __half* __restrict__ in, __half* __restrict__ out, int batchTokens, int framesPerSample, int channelsPerFrame, int patchArea, const __half* __restrict__ temporalWeights){
	const long patchDim = static_cast<long>(channelsPerFrame)*patchArea;
	const long total = static_cast<long>(batchTokens)*patchDim;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	for(long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const long token = idx/patchDim;
		const int feature = static_cast<int>(idx - token*patchDim);
		const int c = feature/patchArea;
		const int patchOffset = feature - c*patchArea;
		float sum = 0.0f;
		const long tokenBase = token*static_cast<long>(framesPerSample)*patchDim;
		for(int t = 0; t < framesPerSample; ++t){
			const long rawIdx = tokenBase + static_cast<long>(t)*patchDim + c*patchArea + patchOffset;
			sum += __half2float(in[rawIdx]) * __half2float(temporalWeights[c*framesPerSample + t]);
		}
		out[idx] = __float2half(sum);
	}
}
void TemporalFusePatches(const __half* in, __half* out, int batchTokens, int framesPerSample, int channelsPerFrame, int patchArea, const __half* temporalWeights){
	const size_t total = static_cast<size_t>(batchTokens)*channelsPerFrame*patchArea;
	if(total == 0) return;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	if(blocks > 0 && tpb > 0) TemporalFusePatchesKernel<<<blocks, tpb>>>(in, out, batchTokens, framesPerSample, channelsPerFrame, patchArea, temporalWeights);
	checkCUDA(cudaGetLastError());
}
__global__ void TemporalUnfusePatchGradsKernel(const __half* __restrict__ fusedGrad, __half* __restrict__ patchGrad, int batchTokens, int framesPerSample, int channelsPerFrame, int patchArea, const __half* __restrict__ temporalWeights){
	const long rawPatchDim = static_cast<long>(framesPerSample)*channelsPerFrame*patchArea;
	const long total = static_cast<long>(batchTokens)*rawPatchDim;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	for(long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const long token = idx/rawPatchDim;
		const int feature = static_cast<int>(idx - token*rawPatchDim);
		const int t = feature/(channelsPerFrame*patchArea);
		const int rem = feature - t*(channelsPerFrame*patchArea);
		const int c = rem/patchArea;
		const int patchOffset = rem - c*patchArea;
		const long fusedIdx = token*static_cast<long>(channelsPerFrame)*patchArea + c*patchArea + patchOffset;
		patchGrad[idx] = __float2half(__half2float(fusedGrad[fusedIdx]) * __half2float(temporalWeights[c*framesPerSample + t]));
	}
}
void TemporalUnfusePatchGrads(const __half* fusedGrad, __half* patchGrad, int batchTokens, int framesPerSample, int channelsPerFrame, int patchArea, const __half* temporalWeights){
	const size_t total = static_cast<size_t>(batchTokens)*framesPerSample*channelsPerFrame*patchArea;
	if(total == 0) return;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	if(blocks > 0 && tpb > 0) TemporalUnfusePatchGradsKernel<<<blocks, tpb>>>(fusedGrad, patchGrad, batchTokens, framesPerSample, channelsPerFrame, patchArea, temporalWeights);
	checkCUDA(cudaGetLastError());
}
__global__ void TemporalWeightGradKernel(const __half* __restrict__ patchBuffer, const __half* __restrict__ fusedGrad, float* __restrict__ gradTemporalWeights, int batchTokens, int framesPerSample, int channelsPerFrame, int patchArea, float scale){
	extern __shared__ float sharedGrad[];
	const int gradCount = channelsPerFrame*framesPerSample;
	for(int i = threadIdx.x; i < gradCount; i += blockDim.x){ sharedGrad[i] = 0.0f; }
	__syncthreads();
	const size_t total = static_cast<size_t>(batchTokens)*patchArea;
	const size_t stride = static_cast<size_t>(blockDim.x)*gridDim.x;
	for(size_t idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int patchOffset = static_cast<int>(idx % patchArea);
		const int token = static_cast<int>(idx/patchArea);
		const long fusedBase = token*static_cast<long>(channelsPerFrame)*patchArea + patchOffset;
		const long rawBase = token*static_cast<long>(framesPerSample)*channelsPerFrame*patchArea + patchOffset;
		for(int c = 0; c < channelsPerFrame; ++c){
			const float fused = __half2float(fusedGrad[fusedBase + c*patchArea]);
			const long channelBase = rawBase + c*patchArea;
			for(int t = 0; t < framesPerSample; ++t){
				atomicAdd(&sharedGrad[c*framesPerSample + t], fused * __half2float(patchBuffer[channelBase + t*static_cast<long>(channelsPerFrame)*patchArea]) * scale);
			}
		}
	}
	__syncthreads();
	for(int i = threadIdx.x; i < gradCount; i += blockDim.x){ atomicAdd(&gradTemporalWeights[i], sharedGrad[i]); }
}
void TemporalWeightGrad(const __half* patchBuffer, const __half* fusedGrad, float* gradTemporalWeights, int batchTokens, int framesPerSample, int channelsPerFrame, int patchArea, bool first, float scale){
	const size_t total = static_cast<size_t>(batchTokens)*patchArea;
	const int gradCount = channelsPerFrame*framesPerSample;
	if(total == 0 || gradCount == 0) return;
	if(first) checkCUDA(cudaMemset(gradTemporalWeights, 0, gradCount*sizeof(float)));
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	if(blocks > 0 && tpb > 0) TemporalWeightGradKernel<<<blocks, tpb, gradCount*sizeof(float)>>>(patchBuffer, fusedGrad, gradTemporalWeights, batchTokens, framesPerSample, channelsPerFrame, patchArea, scale);
	checkCUDA(cudaGetLastError());
}
