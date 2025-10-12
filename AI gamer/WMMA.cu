#define __CUDACC__
#include "CuCommon.cuh"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
namespace{
	constexpr int kMaxTileCols = 128;
	constexpr int kMaxValueBlocks = 8;
	int GetAttentionTileCols(const int T){
		int limited = T;
		if(limited < 16) limited = 16;
		if(limited > kMaxTileCols) limited = kMaxTileCols;
		const int remainder = limited % 16;
		if(remainder != 0) limited += 16 - remainder;
		return limited;
	}
}
__global__ void WmmaAttentionKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ Out, float* __restrict__ AttentionWeights, int B, int T, int D, int H, int tileCols){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	if(rowBlock*16 >= T) return;
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	const int batchHeadOffset = (batch*H + head)*T*D;
	extern __shared__ char sharedMemBytes[];
	auto qShared = reinterpret_cast<__half*>(sharedMemBytes);
	__half* kTiles = qShared + 16*D;
	__half* vTile = kTiles + numWarps*16*16;
	__half* attTile = vTile + 16*16;
	auto scoresTile = reinterpret_cast<float*>(attTile + 16*16);
	float* rowMax = scoresTile + 16*tileCols;
	float* rowSum = rowMax + 16;
	const float scale = rsqrtf(static_cast<float>(D));
	const __half scaleHalf = __float2half(scale);
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
	if(threadIdx.x < 16){
		rowMax[threadIdx.x] = -FLT_MAX;
		rowSum[threadIdx.x] = 0.0f;
	}
	__syncthreads();
#pragma unroll
	for(int d = threadIdx.x; d < D*16; d += blockDim.x){
		const int row = d / D;
		const int col = d % D;
		if(rowBlock*16 + row < T){
			qShared[row*D + col] = __hmul(Q[batchHeadOffset + (rowBlock*16 + row)*D + col], scaleHalf);
		} else{
			qShared[row*D + col] = __float2half(0.0f);
		}
	}
	__syncthreads();
	const int attentionOffset = (batch*H + head)*T*T;
	for(int tileStart = 0; tileStart < T; tileStart += tileCols){
		const int remaining = T - tileStart;
		const int tileWidth = remaining > tileCols ? tileCols : remaining;
		if(tileWidth <= 0) break;
		for(int colBlock = 0; colBlock < tileWidth; colBlock += 16*numWarps){
			const int remainingCols = tileWidth - colBlock;
			int activeWarps = 0;
			if(remainingCols > 0){
				activeWarps = (remainingCols + 15) / 16;
				if(activeWarps > numWarps){ activeWarps = numWarps; }
			}
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpScores;
			if(warpId < activeWarps){ fill_fragment(warpScores, 0.0f); }
			for(int dBlock = 0; dBlock < (D + 15) / 16; dBlock++){
				constexpr int elementsPerTile = 16*16;
				const int totalElements = elementsPerTile * activeWarps;
				for(int idx = threadIdx.x; idx < totalElements; idx += blockDim.x){
					const int warpLocal = idx / elementsPerTile;
					const int tileIndex = idx % elementsPerTile;
					const int col = tileIndex / 16;
					const int row = tileIndex % 16;
					const int localCol = warpLocal*16 + col;
					const int globalCol = tileStart + colBlock + localCol;
					const int globalRow = dBlock*16 + row;
					__half val = __float2half(0.0f);
					if(localCol < remainingCols && globalCol < T && globalRow < D){
						val = K[batchHeadOffset + globalCol*D + globalRow];
					}
					kTiles[warpLocal*elementsPerTile + col*16 + row] = val;
				}
				__syncthreads();
				if(warpId < activeWarps){
					load_matrix_sync(q_frag, qShared + dBlock*16, D);
					load_matrix_sync(k_frag, kTiles + warpId*elementsPerTile, 16);
					mma_sync(warpScores, q_frag, k_frag, warpScores);
				}
				__syncthreads();
			}
			if(warpId < activeWarps){
				store_matrix_sync(scoresTile + colBlock + warpId*16, warpScores, tileCols, wmma::mem_row_major);
			}
			__syncthreads();
		}
		for(int row = warpId; row < 16; row += numWarps){
			const int globalRow = rowBlock*16 + row;
			const float prevMax = rowMax[row];
			const float prevSum = rowSum[row];
			if(globalRow >= T){
				if(laneId == 0){
					rowMax[row] = prevMax;
					rowSum[row] = prevSum;
				}
				continue;
			}
			float localMax = -FLT_MAX;
			for(int col = laneId; col < tileWidth; col += 32){
				const float val = scoresTile[row*tileCols + col];
				localMax = fmaxf(localMax, val);
			}
			for(int offset = 16; offset > 0; offset /= 2){ localMax = fmaxf(localMax, __shfl_down_sync(0xffffffff, localMax, offset)); }
			const float blockMax = __shfl_sync(0xffffffff, localMax, 0);
			const float newMax = fmaxf(prevMax, blockMax);
			const float scalePrev = prevSum > 0.0f ? expf(prevMax - newMax) : 0.0f;
			float localSum = 0.0f;
			for(int col = laneId; col < tileWidth; col += 32){
				const float val = scoresTile[row*tileCols + col];
				localSum += expf(val - newMax);
			}
			for(int offset = 16; offset > 0; offset /= 2){ localSum += __shfl_down_sync(0xffffffff, localSum, offset); }
			if(laneId == 0){
				rowMax[row] = newMax;
				rowSum[row] = prevSum * scalePrev + localSum;
			}
		}
		__syncthreads();
	}
	__syncthreads();
	const int valueBlocks = (D + 15) / 16;
	if(valueBlocks > kMaxValueBlocks){
		if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
			printf("WmmaAttention: value blocks %d exceed limit %d\n", valueBlocks, kMaxValueBlocks);
		}
		return;
	}
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frags[kMaxValueBlocks];
	if(warpId == 0){
		for(int vb = 0; vb < valueBlocks; ++vb){ fill_fragment(out_frags[vb], 0.0f); }
	}
	for(int tileStart = 0; tileStart < T; tileStart += tileCols){
		const int remaining = T - tileStart;
		const int tileWidth = remaining > tileCols ? tileCols : remaining;
		if(tileWidth <= 0) break;
		for(int colBlock = 0; colBlock < tileWidth; colBlock += 16*numWarps){
			const int remainingCols = tileWidth - colBlock;
			int activeWarps = 0;
			if(remainingCols > 0){
				activeWarps = (remainingCols + 15) / 16;
				if(activeWarps > numWarps){ activeWarps = numWarps; }
			}
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpScores;
			if(warpId < activeWarps){ fill_fragment(warpScores, 0.0f); }
			for(int kBlock = 0; kBlock < (D + 15) / 16; kBlock++){
				const int elementsPerTile = 16*16;
				const int totalElements = elementsPerTile * activeWarps;
				for(int idx = threadIdx.x; idx < totalElements; idx += blockDim.x){
					const int warpLocal = idx / elementsPerTile;
					const int tileIndex = idx % elementsPerTile;
					const int col = tileIndex / 16;
					const int row = tileIndex % 16;
					const int localCol = warpLocal*16 + col;
					const int globalCol = tileStart + colBlock + localCol;
					const int globalRow = kBlock*16 + row;
					__half val = __float2half(0.0f);
					if(localCol < remainingCols && globalCol < T && globalRow < D){
						val = K[batchHeadOffset + globalCol*D + globalRow];
					}
					kTiles[warpLocal*elementsPerTile + col*16 + row] = val;
				}
				__syncthreads();
				if(warpId < activeWarps){
					load_matrix_sync(q_frag, qShared + kBlock*16, D);
					load_matrix_sync(k_frag, kTiles + warpId*elementsPerTile, 16);
					mma_sync(warpScores, q_frag, k_frag, warpScores);
				}
				__syncthreads();
			}
			if(warpId < activeWarps){
				store_matrix_sync(scoresTile + colBlock + warpId*16, warpScores, tileCols, wmma::mem_row_major);
			}
			__syncthreads();
		}
		for(int row = warpId; row < 16; row += numWarps){
			const int globalRow = rowBlock*16 + row;
			if(globalRow >= T) continue;
			const float maxVal = rowMax[row];
			const float denom = rowSum[row];
			const float invDenom = denom > 0.0f ? 1.0f / denom : 0.0f;
			for(int col = laneId; col < tileWidth; col += 32){
				const int globalCol = tileStart + col;
				float logits = scoresTile[row*tileCols + col];
				float normalized = 0.0f;
				if(globalCol < T){ normalized = expf(logits - maxVal) * invDenom; }
				scoresTile[row*tileCols + col] = normalized;
				if(AttentionWeights){ AttentionWeights[attentionOffset + globalRow*T + globalCol] = normalized; }
			}
		}
		__syncthreads();
		for(int colBlock = 0; colBlock < tileWidth; colBlock += 16){
			for(int idx = threadIdx.x; idx < 16*16; idx += blockDim.x){
				const int row = idx / 16;
				const int col = idx % 16;
				const int globalRow = rowBlock*16 + row;
				const int localCol = colBlock + col;
				const int globalCol = tileStart + localCol;
				float val = 0.0f;
				if(globalRow < T && localCol < tileWidth && globalCol < T){
					val = scoresTile[row*tileCols + localCol];
				}
				attTile[row*16 + col] = __float2half(val);
			}
			__syncthreads();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> att_frag;
			if(warpId == 0){ load_matrix_sync(att_frag, attTile, 16); }
			for(int vb = 0; vb < valueBlocks; ++vb){
				for(int idx = threadIdx.x; idx < 16*16; idx += blockDim.x){
					const int row = idx / 16;
					const int col = idx % 16;
					const int keyIdx = tileStart + colBlock + row;
					const int valueIdx = vb*16 + col;
					__half val = __float2half(0.0f);
					if(keyIdx < T && valueIdx < D){
						val = V[batchHeadOffset + keyIdx*D + valueIdx];
					}
					vTile[row*16 + col] = val;
				}
				__syncthreads();
				if(warpId == 0){
					load_matrix_sync(v_frag, vTile, 16);
					mma_sync(out_frags[vb], att_frag, v_frag, out_frags[vb]);
				}
				__syncthreads();
			}
		}
	}
	for(int vb = 0; vb < valueBlocks; ++vb){
		if(warpId == 0){
#pragma unroll
			for(int i = 0; i < out_frags[vb].num_elements; i++){
				const int row = i / 16;
				const int col = i % 16;
				const int globalRow = rowBlock*16 + row;
				const int globalCol = vb*16 + col;
				if(globalRow < T && globalCol < D){ Out[batchHeadOffset + globalRow*D + globalCol] = __float2half(out_frags[vb].x[i]); }
			}
		}
		__syncthreads();
	}
}
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, float* AttentionWeights, int B, int T, int D, int H){
	dim3 block(128);
	dim3 grid((T + 15) / 16, B, H);
	const int tileCols = GetAttentionTileCols(T);
	const int warpCount = block.x / 32;
	size_t sharedSize = sizeof(__half)*(16*D + warpCount*16*16 + 2*16*16) + sizeof(float)*(16*tileCols + 16 + 16);
	cudaFuncSetAttribute(WmmaAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	WmmaAttentionKernel<<<grid, block, sharedSize>>>(Q, K, V, Out, AttentionWeights, B, T, D, H, tileCols);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("WmmaAttention Forward error: %s\n", cudaGetErrorString(e));
}
__global__ void ComputeDAttDQKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ attention, float* __restrict__ dAtt, __half* __restrict__ dQ, int B, int T, int D, int H,int tileCols){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	const int rowStart = rowBlock * 16;
	if(rowStart >= T) return;
	const int batchHead = batch * H + head;
	const int embOffset = batchHead * T * D;
	const int attOffset = batchHead * T * T;
	const int numKeyBlocks = (T + 15) / 16;
	const int numDBlocks = (D + 15) / 16;
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	extern __shared__ char sharedBytes[];
	const int elementsPerTile = 16 * 16;
	auto scoreTiles = reinterpret_cast<float*>(sharedBytes);
	float* rowSums = scoreTiles + numWarps * elementsPerTile;
	auto outTile = reinterpret_cast<__half*>(rowSums + 16);
	__half* valueTiles = outTile + elementsPerTile;
	__half* attTiles = valueTiles + numWarps * elementsPerTile;
	if(threadIdx.x < 16){ rowSums[threadIdx.x] = 0.0f; }
	__syncthreads();
	for(int tileStart = 0; tileStart < T; tileStart += tileCols){
		const int remaining = T - tileStart;
		const int tileWidth = remaining > tileCols ? tileCols : remaining;
		if(tileWidth <= 0) break;
		for(int colBlock = 0; colBlock < tileWidth; colBlock += 16 * numWarps){
			const int remainingCols = tileWidth - colBlock;
			if(remainingCols <= 0) break;
			int activeWarps = (remainingCols + 15) / 16;
			if(activeWarps > numWarps) activeWarps = numWarps;
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpScores;
			if(warpId < activeWarps){ fill_fragment(warpScores, 0.0f); }
			for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
				for(int idx = threadIdx.x; idx < elementsPerTile; idx += blockDim.x){
					const int r = idx / 16;
					const int c = idx % 16;
					const int globalRow = rowStart + r;
					const int globalCol = dBlock * 16 + c;
					__half val = __float2half(0.0f);
					if(globalRow < T && globalCol < D){ val = dOut[embOffset + globalRow * D + globalCol]; }
					outTile[r * 16 + c] = val;
				}
				for(int idx = threadIdx.x; idx < activeWarps * elementsPerTile; idx += blockDim.x){
					const int warpLocal = idx / elementsPerTile;
					const int tileIndex = idx % elementsPerTile;
					const int r = tileIndex / 16;
					const int c = tileIndex % 16;
					const int warpColBase = warpLocal * 16;
					const int globalRow = tileStart + colBlock + warpColBase + r;
					const int globalCol = dBlock * 16 + c;
					__half val = __float2half(0.0f);
					if(globalRow < T && globalCol < D && warpColBase + r < remainingCols){
						val = V[embOffset + globalRow * D + globalCol];
					}
					valueTiles[warpLocal * elementsPerTile + c * 16 + r] = val;
				}
				__syncthreads();
				if(warpId < activeWarps){
					wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> outFrag;
					wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> vFrag;
					load_matrix_sync(outFrag, outTile, 16);
					load_matrix_sync(vFrag, valueTiles + warpId * elementsPerTile, 16);
					mma_sync(warpScores, outFrag, vFrag, warpScores);
				}
				__syncthreads();
			}
			if(warpId < activeWarps){ store_matrix_sync(scoreTiles + warpId * elementsPerTile, warpScores, 16, wmma::mem_row_major); }
			__syncthreads();
			for(int idx = threadIdx.x; idx < activeWarps * elementsPerTile; idx += blockDim.x){
				const int warpLocal = idx / elementsPerTile;
				const int tileIndex = idx % elementsPerTile;
				const int r = tileIndex / 16;
				const int c = tileIndex % 16;
				const int warpColBase = warpLocal * 16;
				int validCols = remainingCols - warpColBase;
				if(validCols <= 0) continue;
				if(validCols > 16) validCols = 16;
				const int globalRow = rowStart + r;
				const int globalCol = tileStart + colBlock + warpColBase + c;
				if(globalRow < T && c < validCols && globalCol < T){
					dAtt[attOffset + globalRow * T + globalCol] = scoreTiles[warpLocal * elementsPerTile + r * 16 + c];
				}
			}
			__syncthreads();
			for(int row = warpId; row < 16; row += numWarps){
				const int globalRow = rowStart + row;
				if(globalRow >= T) continue;
				float accum = 0.0f;
				for(int warpLocal = 0; warpLocal < activeWarps; ++warpLocal){
					const int warpColBase = warpLocal * 16;
					int validCols = remainingCols - warpColBase;
					if(validCols <= 0) continue;
					if(validCols > 16) validCols = 16;
					const float* tile = scoreTiles + warpLocal * elementsPerTile + row * 16;
					const int globalBase = tileStart + colBlock + warpColBase;
					for(int c = laneId; c < validCols; c += 32){
						const int globalCol = globalBase + c;
						const float rawVal = tile[c];
						const float attVal = attention[attOffset + globalRow * T + globalCol];
						accum += rawVal * attVal;
					}
				}
				for(int offset = 16; offset > 0; offset >>= 1){ accum += __shfl_down_sync(0xffffffff, accum, offset); }
				if(laneId == 0){ rowSums[row] += accum; }
			}
			__syncthreads();
		}
	}
	__syncthreads();
	for(int tileStart = 0; tileStart < T; tileStart += tileCols){
		const int remaining = T - tileStart;
		const int tileWidth = remaining > tileCols ? tileCols : remaining;
		if(tileWidth <= 0) break;
		for(int colBlock = 0; colBlock < tileWidth; colBlock += 16 * numWarps){
			const int remainingCols = tileWidth - colBlock;
			if(remainingCols <= 0) break;
			int activeWarps = (remainingCols + 15) / 16;
			if(activeWarps > numWarps) activeWarps = numWarps;
			for(int idx = threadIdx.x; idx < activeWarps * elementsPerTile; idx += blockDim.x){
				const int warpLocal = idx / elementsPerTile;
				const int tileIndex = idx % elementsPerTile;
				const int r = tileIndex / 16;
				const int c = tileIndex % 16;
				const int warpColBase = warpLocal * 16;
				int validCols = remainingCols - warpColBase;
				if(validCols <= 0) continue;
				if(validCols > 16) validCols = 16;
				const int globalRow = rowStart + r;
				const int globalCol = tileStart + colBlock + warpColBase + c;
				if(globalRow < T && c < validCols && globalCol < T){
					const float rawVal = dAtt[attOffset + globalRow * T + globalCol];
					const float attVal = attention[attOffset + globalRow * T + globalCol];
					dAtt[attOffset + globalRow * T + globalCol] = attVal * (rawVal - rowSums[r]);
				}
			}
			__syncthreads();
		}
	}
	const float scale = rsqrtf(static_cast<float>(D));
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int keyBlock = warpId; keyBlock < numKeyBlocks; keyBlock += numWarps){
			const int keyBase = keyBlock * 16;
			__half* attTile = attTiles + warpId * elementsPerTile;
			__half* kTile = valueTiles + warpId * elementsPerTile;
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = rowStart + r;
				const int globalCol = keyBase + c;
				float val = 0.0f;
				if(globalRow < T && globalCol < T){ val = dAtt[attOffset + globalRow * T + globalCol]; }
				attTile[r * 16 + c] = __float2half(val);
			}
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = keyBase + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalRow < T && globalCol < D){ val = K[embOffset + globalRow * D + globalCol]; }
				kTile[r * 16 + c] = val;
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> attFrag;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> kFrag;
			load_matrix_sync(attFrag, attTile, 16);
			load_matrix_sync(kFrag, kTile, 16);
			mma_sync(warpAcc, attFrag, kFrag, warpAcc);
			__syncwarp();
		}
		store_matrix_sync(scoreTiles + warpId * elementsPerTile, warpAcc, 16, wmma::mem_row_major);
		__syncthreads();
		for(int idx = threadIdx.x; idx < elementsPerTile; idx += blockDim.x){
			float sum = 0.0f;
			for(int w = 0; w < numWarps; ++w){
				sum += scoreTiles[w * elementsPerTile + idx];
			}
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = rowStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < T && globalCol < D){
				dQ[embOffset + globalRow * D + globalCol] = __float2half(sum * scale);
			}
		}
		__syncthreads();
	}
}
__global__ void ComputeDVKernel(const float* __restrict__ attention, const __half* __restrict__ dOut, __half* __restrict__ dV,int B, int T, int D, int H){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int keyBlock = blockIdx.x;
	const int keyStart = keyBlock * 16;
	if(keyStart >= T) return;
	const int batchHead = batch * H + head;
	const int embOffset = batchHead * T * D;
	const int attOffset = batchHead * T * T;
	const int numRowBlocks = (T + 15) / 16;
	const int numDBlocks = (D + 15) / 16;
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	extern __shared__ char sharedBytes[];
	const int elementsPerTile = 16 * 16;
	auto accumStore = reinterpret_cast<float*>(sharedBytes);
	__half* attTiles = reinterpret_cast<__half*>(accumStore + numWarps * elementsPerTile);
	__half* outTiles = attTiles + numWarps * elementsPerTile;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
			const int queryBase = rowBlock * 16;
			__half* attTile = attTiles + warpId * elementsPerTile;
			__half* outTile = outTiles + warpId * elementsPerTile;
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = queryBase + r;
				const int globalKey = keyStart + c;
				float val = 0.0f;
				if(globalQuery < T && globalKey < T){ val = attention[attOffset + globalQuery * T + globalKey]; }
				attTile[c * 16 + r] = __float2half(val);
			}
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = queryBase + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalQuery < T && globalCol < D){ val = dOut[embOffset + globalQuery * D + globalCol]; }
				outTile[r * 16 + c] = val;
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> outFrag;
			load_matrix_sync(attFrag, attTile, 16);
			load_matrix_sync(outFrag, outTile, 16);
			mma_sync(warpAcc, attFrag, outFrag, warpAcc);
			__syncwarp();
		}
		store_matrix_sync(accumStore + warpId * elementsPerTile, warpAcc, 16, wmma::mem_row_major);
		__syncthreads();
		for(int idx = threadIdx.x; idx < elementsPerTile; idx += blockDim.x){
			float sum = 0.0f;
			for(int w = 0; w < numWarps; ++w){
				sum += accumStore[w * elementsPerTile + idx];
			}
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = keyStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < T && globalCol < D){
				dV[embOffset + globalRow * D + globalCol] = __float2half(sum);
			}
		}
		__syncthreads();
	}
}
__global__ void ComputeDKKernel(const float* __restrict__ dAtt, const __half* __restrict__ Q, __half* __restrict__ dK, int B, int T, int D, int H){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int keyBlock = blockIdx.x;
	const int keyStart = keyBlock * 16;
	if(keyStart >= T) return;
	const int batchHead = batch * H + head;
	const int embOffset = batchHead * T * D;
	const int attOffset = batchHead * T * T;
	const int numRowBlocks = (T + 15) / 16;
	const int numDBlocks = (D + 15) / 16;
	const float scale = rsqrtf(static_cast<float>(D));
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	extern __shared__ char sharedBytes[];
	const int elementsPerTile = 16 * 16;
	auto accumStore = reinterpret_cast<float*>(sharedBytes);
	__half* attTiles = reinterpret_cast<__half*>(accumStore + numWarps * elementsPerTile);
	__half* qTiles = attTiles + numWarps * elementsPerTile;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
			const int queryBase = rowBlock * 16;
			__half* attTile = attTiles + warpId * elementsPerTile;
			__half* qTile = qTiles + warpId * elementsPerTile;
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalKey = keyStart + r;
				const int globalQuery = queryBase + c;
				float val = 0.0f;
				if(globalKey < T && globalQuery < T){ val = dAtt[attOffset + globalQuery * T + globalKey]; }
				attTile[c * 16 + r] = __float2half(val);
			}
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = queryBase + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalQuery < T && globalCol < D){ val = Q[embOffset + globalQuery * D + globalCol]; }
				qTile[r * 16 + c] = val;
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> qFrag;
			load_matrix_sync(attFrag, attTile, 16);
			load_matrix_sync(qFrag, qTile, 16);
			mma_sync(warpAcc, attFrag, qFrag, warpAcc);
			__syncwarp();
		}
		store_matrix_sync(accumStore + warpId * elementsPerTile, warpAcc, 16, wmma::mem_row_major);
		__syncthreads();
		for(int idx = threadIdx.x; idx < elementsPerTile; idx += blockDim.x){
			float sum = 0.0f;
			for(int w = 0; w < numWarps; ++w){
				sum += accumStore[w * elementsPerTile + idx];
			}
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = keyStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < T && globalCol < D){
				dK[embOffset + globalRow * D + globalCol] = __float2half(sum * scale);
			}
		}
		__syncthreads();
	}
}
void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* Att, __half* dQ, __half* dK, __half* dV, float* dAttWorkspace, size_t workspaceElements, int B, int T, int D, int H){
	if(dAttWorkspace == nullptr){
		printf("WmmaAttention Backward workspace pointer is null\n");
		return;
	}
	const size_t requiredElements = static_cast<size_t>(B) * H * T * T;
	if(requiredElements > workspaceElements){
		printf("WmmaAttention Backward requires %zu elements but workspace has %zu (T=%d)\n", requiredElements, workspaceElements, T);
		return;
	}
	const int numRowBlocks = (T + 15) / 16;
	const int numKeyBlocks = (T + 15) / 16;
	const int tileCols = GetAttentionTileCols(T);
	dim3 block(256);
	dim3 gridDQ(numRowBlocks, B, H);
	dim3 gridKV(numKeyBlocks, B, H);
	const int warpCount = block.x / 32;
	const size_t tileElements = 16 * 16;
	const size_t smemDQ = sizeof(float) * (warpCount * tileElements + 16) + sizeof(__half) * (tileElements + 2 * warpCount * tileElements);
	const size_t smemDV = sizeof(float) * (warpCount * tileElements) + sizeof(__half) * (2 * warpCount * tileElements);
	cudaFuncSetAttribute(ComputeDAttDQKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	cudaFuncSetAttribute(ComputeDVKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	cudaFuncSetAttribute(ComputeDKKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	ComputeDAttDQKernel<<<gridDQ, block, smemDQ>>>(Q, K, V, dOut, Att, dAttWorkspace, dQ, B, T, D, H, tileCols);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttention Backward dAtt+dQ error: %s\n", cudaGetErrorString(err));
		return;
	}
	ComputeDVKernel<<<gridKV, block, smemDV>>>(Att, dOut, dV, B, T, D, H);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttention Backward dV error: %s\n", cudaGetErrorString(err));
		return;
	}
	ComputeDKKernel<<<gridKV, block, smemDV>>>(dAttWorkspace, Q, dK, B, T, D, H);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttention Backward dK error: %s\n", cudaGetErrorString(err));
		return;
	}
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess) printf("WmmaAttention Backward sync error: %s\n", cudaGetErrorString(err));
}