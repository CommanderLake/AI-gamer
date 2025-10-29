#include "CuCommon.cuh"
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
__global__ void ExtractPatchesKernelVec2(const __half* __restrict__ x, __half* __restrict__ y, int B, int C, int H, int W, int P){
	const int kPatchArea = P*P;
	const int PH = (H + P - 1) / P;
	const int PW = (W + P - 1) / P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total_elements = static_cast<long>(B)*PH*PW*patchDim;
	const long total_vec2 = total_elements / 2;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	for(long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total_vec2; idx += stride){
		__half2 vals = __float2half2_rn(0.0f);
#pragma unroll 2
		for(int i = 0; i < 2; i++){
			long elem_idx = idx*2 + i;
			if(elem_idx >= total_elements) break;
			long patch = elem_idx / patchDim;
			int inPatch = elem_idx - patch*patchDim;
			int b = patch / (PH*PW);
			int pp = patch - b*(PH*PW);
			int pyPatch = pp / PW;
			int pxPatch = pp - pyPatch*PW;
			int c = inPatch / kPatchArea;
			int rem = inPatch - c*kPatchArea;
			int py = rem / P;
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
	const int PH = (H + P - 1) / P;
	const int PW = (W + P - 1) / P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total = static_cast<long>(B)*PH*PW*patchDim;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	for(long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		long patch = idx / patchDim;
		int inPatch = idx - patch*patchDim;
		int b = patch / (PH*PW);
		int pp = patch - b*(PH*PW);
		int pyPatch = pp / PW;
		int pxPatch = pp - pyPatch*PW;
		int c = inPatch / kPatchArea;
		int rem = inPatch - c*kPatchArea;
		int py = rem / P;
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
	const int PH = (H + P - 1) / P;
	const int PW = (W + P - 1) / P;
	const size_t total = static_cast<size_t>(B)*PH*PW*C*P*P;
	if(total == 0) return;
	if(total % 2 == 0){
		size_t blocks = 0, tpb = 0;
		GetLaunchConfigGridStride(total / 2, blocks, tpb);
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
	const int PH = (H + P - 1) / P;
	const int PW = (W + P - 1) / P;
	const long patchDim = static_cast<long>(C)*kPatchArea;
	const long total = static_cast<long>(B)*PH*PW*patchDim;
	const long stride = static_cast<long>(blockDim.x)*gridDim.x;
	for(long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		long patch = idx / patchDim;
		int inPatch = idx - patch*patchDim;
		int b = patch / (PH*PW);
		int pp = patch - b*(PH*PW);
		int pyPatch = pp / PW;
		int pxPatch = pp - pyPatch*PW;
		int c = inPatch / kPatchArea;
		int rem = inPatch - c*kPatchArea;
		int py = rem / P;
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
	const int PH = (H + P - 1) / P;
	const int PW = (W + P - 1) / P;
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
		const int c = idx / P;
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
__global__ void BuildClassTokenOutputKernel(const __half* __restrict__ classToken, const __half* __restrict__ patches, __half* __restrict__ output, int batchTotal, int embedDim, int numPatches){
	const int tokensWithCls = numPatches + 1;
	const long long total = static_cast<long long>(batchTotal)*tokensWithCls*embedDim;
	const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
	for(long long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int sample = idx / (tokensWithCls*embedDim);
		const int rem = static_cast<int>(idx - static_cast<long long>(sample)*tokensWithCls*embedDim);
		const int token = rem / embedDim;
		const int emb = rem - token*embedDim;
		if(token == 0){
			output[idx] = classToken[emb];
		} else{
			const int patchIndex = sample*numPatches*embedDim + (token - 1)*embedDim + emb;
			output[idx] = patches[patchIndex];
		}
	}
}
void BuildClassTokenOutput(const __half* classToken, const __half* patches, __half* output, int batchTotal, int embedDim, int numPatches){
	const int tokensWithCls = numPatches + 1;
	const long long total = static_cast<long long>(batchTotal)*tokensWithCls*embedDim;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	BuildClassTokenOutputKernel<<<blocks, tpb>>>(classToken, patches, output, batchTotal, embedDim, numPatches);
	checkCUDA(cudaGetLastError());
}
__global__ void StripClassTokenKernel(const __half* __restrict__ input, __half* __restrict__ output, int batchTotal, int embedDim, int numPatches){
	const int tokensWithCls = numPatches + 1;
	const long long total = static_cast<long long>(batchTotal)*numPatches*embedDim;
	const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
	for(long long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int sample = idx / (numPatches*embedDim);
		const int rem = static_cast<int>(idx - static_cast<long long>(sample)*numPatches*embedDim);
		const int token = rem / embedDim;
		const int emb = rem - token*embedDim;
		const long long srcIndex = static_cast<long long>(sample)*tokensWithCls*embedDim + (token + 1)*embedDim + emb;
		output[idx] = input[srcIndex];
	}
}
void StripClassToken(const __half* input, __half* output, int batchTotal, int embedDim, int numPatches){
	const long long total = static_cast<long long>(batchTotal)*numPatches*embedDim;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	StripClassTokenKernel<<<blocks, tpb>>>(input, output, batchTotal, embedDim, numPatches);
	checkCUDA(cudaGetLastError());
}
__global__ void GatherClassTokensKernel(const __half* __restrict__ input, __half* __restrict__ output, int batchTotal, int tokens, int embedDim){
	const long long total = static_cast<long long>(batchTotal)*embedDim;
	const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
	for(long long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int sample = idx / embedDim;
		const int emb = static_cast<int>(idx - static_cast<long long>(sample)*embedDim);
		const long long srcIndex = static_cast<long long>(sample)*tokens*embedDim + emb;
		output[idx] = input[srcIndex];
	}
}
void GatherClassTokens(const __half* input, __half* output, int batchTotal, int tokens, int embedDim){
	const long long total = static_cast<long long>(batchTotal)*embedDim;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	GatherClassTokensKernel<<<blocks, tpb>>>(input, output, batchTotal, tokens, embedDim);
	checkCUDA(cudaGetLastError());
}
__global__ void ScatterClassTokenGradsKernel(const __half* __restrict__ classGrad, __half* __restrict__ output, int batchTotal, int tokens, int embedDim){
	const long long total = static_cast<long long>(batchTotal)*tokens*embedDim;
	const long long stride = static_cast<long long>(blockDim.x)*gridDim.x;
	for(long long idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int sample = idx / (tokens*embedDim);
		const int rem = static_cast<int>(idx - static_cast<long long>(sample)*tokens*embedDim);
		const int token = rem / embedDim;
		const int emb = rem - token*embedDim;
		if(token == 0){
			const int srcIndex = sample*embedDim + emb;
			output[idx] = classGrad[srcIndex];
		} else{ output[idx] = __float2half(0.0f); }
	}
}
void ScatterClassTokenGrads(const __half* classGrad, __half* output, int batchTotal, int tokens, int embedDim){
	const auto total = static_cast<long long>(batchTotal)*tokens*embedDim;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	ScatterClassTokenGradsKernel<<<blocks, tpb>>>(classGrad, output, batchTotal, tokens, embedDim);
	checkCUDA(cudaGetLastError());
}
__global__ void SumClassTokenGradKernel(const __half* __restrict__ grad, __half* __restrict__ out, int batchTotal, int embedDim, int tokens, bool first, float scale){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < embedDim; idx += stride){
		float sum = 0.0f;
		for(int b = 0; b < batchTotal; ++b){
			const int gradIndex = (b*tokens)*embedDim + idx;
			sum += __half2float(grad[gradIndex]);
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
void SumClassTokenGrad(const __half* grad, __half* out, int batchTotal, int embedDim, int tokens, bool first, float scale){
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(embedDim, blocks, tpb);
	SumClassTokenGradKernel<<<blocks, tpb>>>(grad, out, batchTotal, embedDim, tokens, first, scale);
	checkCUDA(cudaGetLastError());
}