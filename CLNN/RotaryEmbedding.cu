#define __CUDACC__
#include "CuCommon.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
__global__ void RotaryEmbeddingKernel(__half* __restrict__ out, const int* __restrict__ positionOffsets, int batchSize, int tokens, int embedDim, int numHeads, int rotaryDim, int basePosition, float theta, bool interleaved, bool inverse){
	const int headDim = embedDim/numHeads;
	const int halfRotary = rotaryDim/2;
	const int totalPairs = batchSize*tokens*numHeads*halfRotary;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < totalPairs; idx += blockDim.x*gridDim.x){
		int v = idx;
		const int pair = v%halfRotary;
		v /= halfRotary;
		const int head = v%numHeads;
		v /= numHeads;
		const int token = v%tokens;
		const int batch = v/tokens;
		const int offset = positionOffsets ? positionOffsets[batch] : 0;
		const int position = basePosition + offset + token;
		const float invFreq = expf(-logf(theta)*static_cast<float>(2*pair)/static_cast<float>(rotaryDim));
		float s, c;
		sincosf(static_cast<float>(position)*invFreq, &s, &c);
		if(inverse){ s = -s; }
		const int d0 = interleaved ? 2*pair : pair;
		const int d1 = interleaved ? 2*pair + 1 : pair + halfRotary;
		const int col = batch*tokens + token;
		const int rowBase = head*headDim;
		const size_t idx0 = static_cast<size_t>(col)*embedDim + rowBase + d0;
		const size_t idx1 = static_cast<size_t>(col)*embedDim + rowBase + d1;
		const float x0 = __half2float(out[idx0]);
		const float x1 = __half2float(out[idx1]);
		out[idx0] = __float2half(fmaf(-x1, s, x0*c));
		out[idx1] = __float2half(fmaf(x0, s, x1*c));
	}
}
void RotaryEmbeddingForward(__half* out, const __half* in, const int* positionOffsets, int batchSize, int tokens, int embedDim, int numHeads, int rotaryDim, int basePosition, float theta, bool interleaved, bool inverse){
	if(!out || !in){
		fprintf(stderr, "RotaryEmbeddingForward: Null pointer input\n");
		return;
	}
	if(batchSize <= 0 || tokens <= 0 || embedDim <= 0 || numHeads <= 0 || embedDim%numHeads != 0 || theta <= 1.0f){
		fprintf(stderr, "RotaryEmbeddingForward: Invalid dimensions batch=%d, tokens=%d, embedDim=%d, heads=%d, theta=%f\n", batchSize, tokens, embedDim, numHeads, theta);
		return;
	}
	const int headDim = embedDim/numHeads;
	if(rotaryDim <= 0){ rotaryDim = headDim; }
	if(rotaryDim%2 != 0 || rotaryDim > headDim){
		fprintf(stderr, "RotaryEmbeddingForward: Invalid rotaryDim=%d for headDim=%d\n", rotaryDim, headDim);
		return;
	}
	const size_t total = static_cast<size_t>(batchSize)*tokens*embedDim;
	if(out != in){ checkCUDA(cudaMemcpy(out, in, total*sizeof(__half), cudaMemcpyDeviceToDevice)); }
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(static_cast<size_t>(batchSize)*tokens*numHeads*(rotaryDim/2), blocks, tpb);
	RotaryEmbeddingKernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(tpb)>>>(out, positionOffsets, batchSize, tokens, embedDim, numHeads, rotaryDim, basePosition, theta, interleaved, inverse);
	checkCUDA(cudaGetLastError());
}
