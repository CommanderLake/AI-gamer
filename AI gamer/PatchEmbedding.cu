#include "CuCommon.cuh"
#include <cstdio>
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
		if(blocks > 0 && tpb > 0)
			ExtractPatchesKernelVec2<<<blocks, tpb>>>(in, out, B, C, H, W, P);
	} else{
		size_t blocks = 0, tpb = 0;
		GetLaunchConfigGridStride(total, blocks, tpb);
		if(blocks > 0 && tpb > 0)
			ExtractPatchesKernel<<<blocks, tpb>>>(in, out, B, C, H, W, P);
	}
	const auto e = cudaGetLastError();
	if(e != cudaSuccess)
		printf("ExtractPatches error: %s\n", cudaGetErrorString(e));
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
		// For non-overlapping patches, direct assignment is correct
		// If you need overlapping patches, use atomicAdd
		dx[dstIdx] = dy[idx];
	}
}
void CombinePatchGrads(const __half* dy, __half* dx, int B, int C, int H, int W, int P){
	cudaMemset(dx, 0, B*C*H*W*sizeof(__half));
	const int PH = (H + P - 1)/P;
	const int PW = (W + P - 1)/P;
	const size_t total = static_cast<size_t>(B)*PH*PW*C*P*P;
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(total, blocks, tpb);
	CombinePatchGradsKernel<<<blocks, tpb>>>(dy, dx, B, C, H, W, P);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("CombinePatchGrads error: %s\n", cudaGetErrorString(e));
}
__global__ void SumPositionalGradKernel(const __half* grad, __half* out, int batchTotal, int seqLength, int C, int P, bool first, float scale){
	const int featureCount = C*P;
	const int total = seqLength*featureCount;
	const int stride = blockDim.x*gridDim.x;
	const int sequences = batchTotal/seqLength;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int cp = idx % featureCount;
		const int t = idx/featureCount;
		const int c = cp / P;
		const int p = cp - c*P;
		float sum = 0.0f;
		for(int s = 0; s < sequences; ++s){
			const int sample = s*seqLength + t;
			const int gradIndex = c + C*(sample*P + p);
			sum += __half2float(grad[gradIndex]);
		}
		const float scaled = sum*scale;
		const int outIndex = c + C*(t*P + p);
		if(first){
			out[outIndex] = __float2half(scaled);
		} else{
			const float prev = __half2float(out[outIndex]);
			out[outIndex] = __float2half(prev + scaled);
		}
	}
}

void SumPositionalGrad(const __half* grad, __half* out, int batchTotal, int seqLength, int C, int P, bool first, float scale){
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(seqLength*C*P, blocks, tpb);
	SumPositionalGradKernel<<<blocks, tpb>>>(grad, out, batchTotal, seqLength, C, P, first, scale);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("SumPositionalGrad error: %s\n", cudaGetErrorString(e));
}

__global__ void AddTemporalPositionalEmbeddingKernel(__half* output, const __half* posEmbed, int batchTotal, int seqLength, int featureSize){
	const int stride = blockDim.x*gridDim.x;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < batchTotal*featureSize; idx += stride){
		const int sample = idx/featureSize;
		const int t = seqLength > 0 ? sample % seqLength : 0;
		const int posIndex = t*featureSize + (idx - sample*featureSize);
		output[idx] = __hadd(output[idx], posEmbed[posIndex]);
	}
}
void AddTemporalPositionalEmbedding(__half* output, const __half* posEmbed, int batchTotal, int seqLength, int featureSize){
	if(seqLength <= 0){ return; }
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(batchTotal*featureSize, blocks, tpb);
	AddTemporalPositionalEmbeddingKernel<<<blocks, tpb>>>(output, posEmbed, batchTotal, seqLength, featureSize);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("AddTemporalPositionalEmbedding error: %s\n", cudaGetErrorString(e));
}
constexpr int MAX_TEMPORAL_SEQ = 16;
__global__ void TemporalBlendForwardKernel(const __half* input, __half* output, int batch, int seqLength, int featureSize){
	const int stride = blockDim.x*gridDim.x;
	const int total = batch*featureSize;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int b = idx/featureSize;
		const int f = idx - b*featureSize;
		float values[MAX_TEMPORAL_SEQ];
		float sum = 0.0f;
		for(int t = 0; t < seqLength; ++t){
			const int offset = ((b*seqLength + t)*featureSize) + f;
			const float val = __half2float(input[offset]);
			values[t] = val;
			sum += val;
		}
		const float mean = sum/static_cast<float>(seqLength);
		for(int t = 0; t < seqLength; ++t){
			const float prev = t == 0 ? values[t] : values[t - 1];
			const float diff = values[t] - prev;
			const float blended = 0.5f*(values[t] + mean) + 0.5f*diff;
			const int outIndex = ((b*seqLength + t)*featureSize) + f;
			output[outIndex] = __float2half(blended);
		}
	}
}
void TemporalBlendForward(const __half* input, __half* output, int batch, int seqLength, int featureSize){
	if(seqLength <= 0 || batch <= 0){ return; }
	if(seqLength > MAX_TEMPORAL_SEQ){
		checkCUDA(cudaMemcpy(output, input, static_cast<size_t>(batch)*seqLength*featureSize*sizeof(__half), cudaMemcpyDeviceToDevice));
		return;
	}
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(batch*featureSize, blocks, tpb);
	TemporalBlendForwardKernel<<<blocks, tpb>>>(input, output, batch, seqLength, featureSize);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("TemporalBlendForward error: %s\n", cudaGetErrorString(e));
}
__global__ void TemporalBlendBackwardKernel(const __half* gradOut, __half* gradIn, int batch, int seqLength, int featureSize){
	const int stride = blockDim.x*gridDim.x;
	const int total = batch*featureSize;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += stride){
		const int b = idx/featureSize;
		const int f = idx - b*featureSize;
		float grads[MAX_TEMPORAL_SEQ];
		float accum[MAX_TEMPORAL_SEQ];
		float sum = 0.0f;
		for(int t = 0; t < seqLength; ++t){
			const int offset = ((b*seqLength + t)*featureSize) + f;
			const float val = __half2float(gradOut[offset]);
			grads[t] = val;
			sum += val;
			accum[t] = 0.0f;
		}
		const float meanContribution = 0.5f*sum/static_cast<float>(seqLength);
		for(int t = 0; t < seqLength; ++t){
			accum[t] += meanContribution;
			if(t == 0){
				accum[0] += 0.5f*grads[0];
			} else{
				accum[t] += grads[t];
				accum[t - 1] -= 0.5f*grads[t];
			}
			if(t < seqLength - 1){ accum[t] -= 0.5f*grads[t + 1]; }
		}
		for(int t = 0; t < seqLength; ++t){
			const int outIndex = ((b*seqLength + t)*featureSize) + f;
			gradIn[outIndex] = __float2half(accum[t]);
		}
	}
}
void TemporalBlendBackward(const __half* gradOut, __half* gradIn, int batch, int seqLength, int featureSize){
	if(seqLength <= 0 || batch <= 0){ return; }
	if(seqLength > MAX_TEMPORAL_SEQ){
		checkCUDA(cudaMemcpy(gradIn, gradOut, static_cast<size_t>(batch)*seqLength*featureSize*sizeof(__half), cudaMemcpyDeviceToDevice));
		return;
	}
	size_t blocks = 0, tpb = 0;
	GetLaunchConfigGridStride(batch*featureSize, blocks, tpb);
	TemporalBlendBackwardKernel<<<blocks, tpb>>>(gradOut, gradIn, batch, seqLength, featureSize);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("TemporalBlendBackward error: %s\n", cudaGetErrorString(e));
}