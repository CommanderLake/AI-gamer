#define __CUDACC__
#include "CuCommon.cuh"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <algorithm>
using namespace nvcuda;
namespace{
	// Configuration constants with safer defaults
	constexpr int kMaxTileCols = 128;
	constexpr int kMaxValueBlocks = 32;
	constexpr int kSharedMemPad = 8;
	constexpr float SOFTMAX_FTZ_THRESHOLD = -12.0f;
	constexpr float SOFTMAX_MAX_INPUT = 20.0f;
	constexpr size_t kMaxSharedMemory = 98304;
	constexpr int kMinTokens = 1;
	constexpr int kMaxTokens = 8192;
	constexpr int kMaxBatch = 4096;
	constexpr int kMaxHeadDim = 512;
	constexpr int kWarpSize = 32;
	constexpr int kTileSize = 16;
	constexpr int kDefaultThreads = 128;
	// Helper for ceiling division
	__host__ __device__ inline int DivCeil(int a, int b){ return (a + b - 1)/b; }
	// Get optimal tile columns based on token count
	__host__ int GetAttentionTileCols(const int T){
		int limited = T;
		if(limited < 16) limited = 16;
		if(limited > kMaxTileCols) limited = kMaxTileCols;
		// Round up to multiple of 16 for WMMA
		const int remainder = limited % 16;
		if(remainder != 0) limited += 16 - remainder;
		// Use smaller tiles for very long sequences to save shared memory
		if(T > 2048 && limited > 64){ limited = 64; }
		return limited;
	}
	// Validate dimensions before kernel launch
	__host__ bool ValidateAttentionDimensions(int batchSize, int tokens, int headDim, int heads, size_t& sharedMemRequired){
		if(batchSize <= 0 || batchSize > kMaxBatch){
			printf("Invalid batch size: %d (must be 1-%d)\n", batchSize, kMaxBatch);
			return false;
		}
		if(tokens < kMinTokens || tokens > kMaxTokens){
			printf("Invalid token count: %d (must be %d-%d)\n", tokens, kMinTokens, kMaxTokens);
			return false;
		}
		if(headDim <= 0 || headDim > kMaxHeadDim){
			printf("Invalid head dimension: %d (must be 1-%d)\n", headDim, kMaxHeadDim);
			return false;
		}
		if(heads <= 0 || heads > 128){
			printf("Invalid head count: %d (must be 1-128)\n", heads);
			return false;
		}
		if(headDim % 16 != 0){ printf("Warning: headDim=%d not multiple of 16, padding will be applied\n", headDim); }
		const int qBlocks = (headDim + 15)/16;
		if(qBlocks > kMaxValueBlocks){
			printf("Head dimension %d requires %d blocks, exceeds limit %d\n", headDim, qBlocks, kMaxValueBlocks);
			return false;
		}
		// Calculate shared memory requirement
		const int tileCols = GetAttentionTileCols(tokens);
		const int warpCount = kDefaultThreads/32;
		const int qStride = (headDim + 15)/16*16 + kSharedMemPad;
		const int tileStride = 16 + kSharedMemPad;
		const int valueBlocks = (headDim + 15)/16;
		if(valueBlocks > kMaxValueBlocks){
			printf("Value blocks %d exceed limit %d\n", valueBlocks, kMaxValueBlocks);
			return false;
		}
		const int valueStride = valueBlocks*16;
		sharedMemRequired = sizeof(__half)*(16*qStride + warpCount*tileStride*16 + tileStride*16) + sizeof(float)*(16*tileCols + 48 + 16*valueStride);
		if(sharedMemRequired > kMaxSharedMemory){
			printf("Required shared memory %zu exceeds limit %zu (tokens=%d, headDim=%d)\n", sharedMemRequired, kMaxSharedMemory, tokens, headDim);
			return false;
		}
		return true;
	}
	// Safe warp reduction operations
	template <typename T> __device__ __forceinline__ T WarpReduceSum(T val){
#pragma unroll
		for(int offset = 16; offset > 0; offset /= 2){ val += __shfl_xor_sync(0xffffffff, val, offset); }
		return val;
	}
	template <typename T> __device__ __forceinline__ T WarpReduceMax(T val){
#pragma unroll
		for(int offset = 16; offset > 0; offset /= 2){ val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset)); }
		return val;
	}
	__device__ __forceinline__ void ReduceStoreTile16(const float* tileAccum, int tileStride, int warpCount, int paddedTileElements, __half* __restrict__ dest, size_t destBaseOffset, int rowBase, int colBase, int rowLimit, int colLimit, int rowStride, size_t totalElements, float scale = 1.0f){
#pragma unroll 4
		for(int idx = threadIdx.x; idx < 16*16; idx += blockDim.x){
			const int r = idx/16;
			const int c = idx % 16;
			float sum = 0.0f;
#pragma unroll
			for(int w = 0; w < warpCount; ++w){
				const size_t tileOffset = static_cast<size_t>(w)*static_cast<size_t>(paddedTileElements) + static_cast<size_t>(r)*tileStride + c;
				sum += tileAccum[tileOffset];
			}
			const int globalRow = rowBase + r;
			const int globalCol = colBase + c;
			if(globalRow < rowLimit && globalCol < colLimit){
				const size_t destIdx = destBaseOffset + static_cast<size_t>(globalRow)*rowStride + globalCol;
				if(destIdx < totalElements){ dest[destIdx] = __float2half(sum*scale); }
			}
		}
	}
}
// ============================================================================
// FORWARD KERNEL
// ============================================================================
__global__ void WmmaAttentionKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ Out, __half* __restrict__ AttentionWeights, const float* __restrict__ AttentionMask, const float* __restrict__ RelPosBias, const int* __restrict__ RelPosIndex, int relPosSize, int batchSize, int tokens, int headDim, int heads, int tileCols, int maskBatchSize, int maskHeads){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	// Early exit for out-of-bounds blocks
	if(batch >= batchSize || head >= heads || rowBlock*16 >= tokens) return;
	const int warpId = threadIdx.x/32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x/32;
	// Validate warp configuration
	if(numWarps == 0 || numWarps > 8) return;
	const size_t batchHeadOffset = (static_cast<size_t>(batch)*heads + head)*tokens*headDim;
	const size_t totalElements = static_cast<size_t>(batchSize)*heads*tokens*headDim;
	extern __shared__ char sharedMemBytes[];
	// Calculate strides with alignment
	const int qStride = ((headDim + 15)/16*16) + kSharedMemPad;
	const int tileStride = 16 + kSharedMemPad;
	const int valueBlocks = (headDim + 15)/16;
	const int valueStride = valueBlocks*16;
	auto qShared = reinterpret_cast<__half*>(sharedMemBytes);
	__half* warpTiles = qShared + 16*qStride;
	__half* attTile = warpTiles + numWarps*tileStride*16;
	auto scoresTile = reinterpret_cast<float*>(attTile + tileStride*16);
	float* rowMax = scoresTile + 16*tileCols;
	float* rowSum = rowMax + 16;
	float* rowScale = rowSum + 16;
	float* outAccum = rowScale + 16;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	const __half scaleHalf = __float2half(scale);
	const __half2 scaleHalf2 = __halves2half2(scaleHalf, scaleHalf);
	const int qBlocks = (headDim + 15)/16;
	// Runtime validation
	if(qBlocks > kMaxValueBlocks || qBlocks <= 0) return;
	if(valueBlocks > kMaxValueBlocks || valueBlocks <= 0) return;
	// WMMA fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
	// Initialize row statistics
	if(threadIdx.x < 16){
		const int row = threadIdx.x;
		const int globalRow = rowBlock*16 + row;
		rowMax[row] = -1e20f; // Use large but not infinite value
		rowSum[row] = 0.0f;
		rowScale[row] = 0.0f;
	}
	__syncthreads();
	// Load and scale Q matrix
	for(int row = 0; row < 16; ++row){
		const int globalRow = rowBlock*16 + row;
		__half* sharedRow = qShared + row*qStride;
		// Initialize shared memory
		for(int col = threadIdx.x; col < qStride; col += blockDim.x){ sharedRow[col] = __float2half(0.0f); }
		__syncthreads();
		if(globalRow < tokens){
			const size_t qRowOffset = batchHeadOffset + globalRow*headDim;
			const __half* rowSrc = Q + qRowOffset;
			__half* rowDst = sharedRow;
			const int vecCount = headDim/2;
			auto srcVec = reinterpret_cast<const __half2*>(rowSrc);
			auto dstVec = reinterpret_cast<__half2*>(rowDst);
			for(int vec = threadIdx.x; vec < vecCount; vec += blockDim.x){
				const size_t baseIdx = static_cast<size_t>(vec)*2;
				const size_t maxIdx = qRowOffset + baseIdx + 1;
				if(maxIdx < totalElements){
					dstVec[vec] = __hmul2(srcVec[vec], scaleHalf2);
				} else if(qRowOffset + baseIdx < totalElements){
					const __half first = __hmul(rowSrc[baseIdx], scaleHalf);
					dstVec[vec] = __halves2half2(first, __float2half(0.0f));
				}
			}
			if((headDim & 1) && threadIdx.x == 0){
				const size_t tailIdx = qRowOffset + headDim - 1;
				if(tailIdx < totalElements){ rowDst[headDim - 1] = __hmul(rowSrc[headDim - 1], scaleHalf); } else{ rowDst[headDim - 1] = __float2half(0.0f); }
			}
		}
	}
	__syncthreads();
	// Initialize output accumulation buffer
	for(int idx = threadIdx.x; idx < 16*valueStride; idx += blockDim.x){ outAccum[idx] = 0.0f; }
	__syncthreads();
	const size_t attentionOffset = (static_cast<size_t>(batch)*heads + head)*tokens*tokens;
	const size_t maxAttentionIdx = static_cast<size_t>(batchSize)*heads*tokens*tokens;
	const int effectiveMaskBatchSize = maskBatchSize > 0 ? maskBatchSize : batchSize;
	const int effectiveMaskHeads = maskHeads > 0 ? maskHeads : heads;
	const int maskBatchIndex = batch % effectiveMaskBatchSize;
	const int maskHeadIndex = head % effectiveMaskHeads;
	const size_t maskOffset = (static_cast<size_t>(maskBatchIndex)*effectiveMaskHeads + maskHeadIndex)*tokens*tokens;
	const bool hasRelPosBias = (RelPosBias != nullptr && RelPosIndex != nullptr && relPosSize > 0);
	const size_t relPosBase = static_cast<size_t>(head)*relPosSize;
	// Process K tiles - compute QK^T
	for(int tileStart = 0; tileStart < tokens; tileStart += tileCols){
		const int remaining = (tokens - tileStart < tileCols) ? (tokens - tileStart) : tileCols;
		if(remaining <= 0) break;
		// Process columns in blocks of 16*numWarps
		for(int colBlock = 0; colBlock < remaining; colBlock += 16*numWarps){
			const int remainingCols = (remaining - colBlock < 16*numWarps) ? (remaining - colBlock) : (16*numWarps);
			const int activeWarps = (remainingCols + 15)/16;
			if(activeWarps <= 0 || activeWarps > numWarps) continue;
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpScores;
			if(warpId < activeWarps){ fill_fragment(warpScores, 0.0f); }
			// Process K blocks
			for(int kBlock = 0; kBlock < qBlocks; kBlock++){
				if(kBlock*16 >= headDim) break;
				// Collaborative K tile loading
				const int rowPairs = 8;
				const int vectorsPerWarp = 16*rowPairs;
				const int totalVectors = vectorsPerWarp*activeWarps;
				for(int vec = threadIdx.x; vec < totalVectors; vec += blockDim.x){
					const int warpLocal = vec/vectorsPerWarp;
					const int warpOffset = vec % vectorsPerWarp;
					const int col = warpOffset/rowPairs;
					const int pair = warpOffset % rowPairs;
					const int row = pair*2;
					const int localCol = warpLocal*16 + col;
					const int globalCol = tileStart + colBlock + localCol;
					const int baseIdx = warpLocal*tileStride*16 + col*tileStride + row;
					__half first = __float2half(0.0f);
					__half second = __float2half(0.0f);
					const bool validCol = (warpLocal < activeWarps) && (localCol < remainingCols) && (globalCol < tokens);
					if(validCol){
						const int globalRow0 = kBlock*16 + row;
						const int globalRow1 = globalRow0 + 1;
						if(globalRow0 < headDim){
							const size_t idx0 = batchHeadOffset + static_cast<size_t>(globalCol)*headDim + globalRow0;
							if(idx0 < totalElements){ first = K[idx0]; }
						}
						if(globalRow1 < headDim){
							const size_t idx1 = batchHeadOffset + static_cast<size_t>(globalCol)*headDim + globalRow1;
							if(idx1 < totalElements){ second = K[idx1]; }
						}
					}
					reinterpret_cast<__half2*>(warpTiles + baseIdx)[0] = __halves2half2(first, second);
				}
				__syncthreads();
				if(warpId < activeWarps){
					load_matrix_sync(q_frag, qShared + kBlock*16, qStride);
					load_matrix_sync(k_frag, warpTiles + warpId*tileStride*16, tileStride);
					mma_sync(warpScores, q_frag, k_frag, warpScores);
				}
				__syncthreads();
			}
			if(warpId < activeWarps){ store_matrix_sync(scoresTile + colBlock + warpId*16, warpScores, tileCols, wmma::mem_row_major); }
		}
		__syncthreads();
		if(AttentionMask != nullptr){
#pragma unroll 4
			for(int row = warpId; row < 16; row += numWarps){
				const int globalRow = rowBlock*16 + row;
				if(globalRow >= tokens) continue;
#pragma unroll 4
				for(int col = laneId; col < remaining; col += 32){
					const int globalCol = tileStart + col;
					if(globalCol < tokens){
						const size_t maskIdx = maskOffset + static_cast<size_t>(globalRow)*tokens + globalCol;
						scoresTile[row*tileCols + col] += AttentionMask[maskIdx];
					}
				}
			}
			__syncthreads();
		}
		if(hasRelPosBias){
#pragma unroll 4
			for(int row = warpId; row < 16; row += numWarps){
				const int globalRow = rowBlock*16 + row;
				if(globalRow >= tokens) continue;
#pragma unroll 4
				for(int col = laneId; col < remaining; col += 32){
					const int globalCol = tileStart + col;
					if(globalCol < tokens){
						const size_t relIdx = static_cast<size_t>(RelPosIndex[globalRow*tokens + globalCol]);
						scoresTile[row*tileCols + col] += RelPosBias[relPosBase + relIdx];
					}
				}
			}
			__syncthreads();
		}
		// Compute softmax statistics
#pragma unroll 4
		for(int row = warpId; row < 16; row += numWarps){
			const int globalRow = rowBlock*16 + row;
			if(globalRow >= tokens) continue;
			const float prevMax = rowMax[row];
			const float prevSum = rowSum[row];
			// Find max
			float localMax = -1e20f;
#pragma unroll 4
			for(int col = laneId; col < remaining; col += 32){
				const int globalCol = tileStart + col;
				if(globalCol < tokens){
					const float val = scoresTile[row*tileCols + col];
					localMax = fmaxf(localMax, val);
				}
			}
			localMax = WarpReduceMax(localMax);
			const float newMax = fmaxf(prevMax, localMax);
			// Compute exp sum with stability
			float scalePrev = 0.0f;
			if(prevSum > 0.0f){
				const float diff = prevMax - newMax;
				if(diff > -20.0f && diff < 20.0f){ scalePrev = expf(diff); }
			}
			float localSum = 0.0f;
#pragma unroll 4
			for(int col = laneId; col < remaining; col += 32){
				const int globalCol = tileStart + col;
				if(globalCol < tokens){
					const float val = scoresTile[row*tileCols + col];
					const float diff = val - newMax;
					if(diff > SOFTMAX_FTZ_THRESHOLD && diff < SOFTMAX_MAX_INPUT){ localSum += expf(diff); }
				}
			}
			localSum = WarpReduceSum(localSum);
			if(laneId == 0){
				rowMax[row] = newMax;
				rowSum[row] = prevSum*scalePrev + localSum;
				rowScale[row] = scalePrev;
			}
		}
		__syncthreads();
		// Rescale prior output accumulation to match new max.
		for(int idx = threadIdx.x; idx < 16*valueStride; idx += blockDim.x){
			const int row = idx/valueStride;
			outAccum[idx] *= rowScale[row];
		}
		__syncthreads();
		// Compute exp(logits - newMax) for this tile.
#pragma unroll 4
		for(int row = warpId; row < 16; row += numWarps){
			const int globalRow = rowBlock*16 + row;
			if(globalRow >= tokens) continue;
			const float maxVal = rowMax[row];
			const float sumVal = rowSum[row];
			const float invSum = (sumVal > 1e-10f) ? (1.0f/sumVal) : 0.0f;
#pragma unroll 4
			for(int col = laneId; col < remaining; col += 32){
				const int globalCol = tileStart + col;
				if(globalCol < tokens){
					const float logit = scoresTile[row*tileCols + col];
					const float diff = logit - maxVal;
					float expVal = 0.0f;
					if(diff > SOFTMAX_FTZ_THRESHOLD && diff < SOFTMAX_MAX_INPUT){ expVal = expf(diff); }
					scoresTile[row*tileCols + col] = expVal;
					if(AttentionWeights != nullptr){
						float normalized = 0.0f;
						if(isfinite(invSum)){ normalized = expVal*invSum; }
						normalized = fminf(fmaxf(normalized, 0.0f), 1.0f);
						const size_t attIdx = attentionOffset + globalRow*tokens + globalCol;
						if(attIdx < maxAttentionIdx){ AttentionWeights[attIdx] = __float2half(normalized); }
					}
				}
			}
		}
		__syncthreads();
		// Matrix multiply exp(logits) with V
		for(int colBlock = 0; colBlock < remaining; colBlock += 16){
			if(colBlock >= remaining) break;
			// Load attention tile
#pragma unroll 4
			for(int idx = threadIdx.x; idx < 16*16; idx += blockDim.x){
				const int row = idx/16;
				const int col = idx % 16;
				const int localCol = colBlock + col;
				const int globalCol = tileStart + localCol;
				float val = 0.0f;
				if(rowBlock*16 + row < tokens && localCol < remaining && globalCol < tokens){ val = scoresTile[row*tileCols + localCol]; }
				attTile[row*tileStride + col] = __float2half(val);
			}
			__syncthreads();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> att_frag;
			load_matrix_sync(att_frag, attTile, tileStride);
			// Process V blocks
			for(int vb = 0; vb < valueBlocks; ++vb){
				if(vb*16 >= headDim) break;
				// Load V tile
				const int colPairs = 8;
				const int totalPairs = 16*colPairs;
				for(int pairIdx = threadIdx.x; pairIdx < totalPairs; pairIdx += blockDim.x){
					const int row = pairIdx/colPairs;
					const int pair = pairIdx % colPairs;
					const int col = pair*2;
					const int keyIdx = tileStart + colBlock + row;
					const int baseIdx = row*tileStride + col;
					__half first = __float2half(0.0f);
					__half second = __float2half(0.0f);
					if(row < remaining && keyIdx < tokens){
						const int valueIdx0 = vb*16 + col;
						if(valueIdx0 < headDim){
							const size_t vIdx0 = batchHeadOffset + static_cast<size_t>(keyIdx)*headDim + valueIdx0;
							if(vIdx0 < totalElements){ first = V[vIdx0]; }
						}
						const int valueIdx1 = valueIdx0 + 1;
						if(valueIdx1 < headDim){
							const size_t vIdx1 = batchHeadOffset + static_cast<size_t>(keyIdx)*headDim + valueIdx1;
							if(vIdx1 < totalElements){ second = V[vIdx1]; }
						}
					}
					reinterpret_cast<__half2*>(warpTiles + baseIdx)[0] = __halves2half2(first, second);
				}
				__syncthreads();
				if(warpId < numWarps && (vb % numWarps) == warpId){
					load_matrix_sync(v_frag, warpTiles, tileStride);
					wmma::fragment<wmma::accumulator, 16, 16, 16, float> outFrag;
					fill_fragment(outFrag, 0.0f);
					mma_sync(outFrag, att_frag, v_frag, outFrag);
					for(int i = 0; i < outFrag.num_elements; ++i){
						const int row = i/16;
						const int col = i % 16;
						const int outRow = row;
						const int outCol = vb*16 + col;
						if(outRow < 16 && outCol < headDim){
							const int outIdx = outRow*valueStride + outCol;
							outAccum[outIdx] += outFrag.x[i];
						}
					}
				}
				__syncthreads();
			}
		}
	}
	__syncthreads();
	// Store output
	for(int idx = threadIdx.x; idx < 16*valueStride; idx += blockDim.x){
		const int row = idx/valueStride;
		const int col = idx % valueStride;
		const int globalRow = rowBlock*16 + row;
		const int globalCol = col;
		const float sumVal = rowSum[row];
		const float invSum = (sumVal > 1e-10f) ? (1.0f/sumVal) : 0.0f;
		if(row < 16 && col < headDim && globalRow < tokens && globalCol < headDim){
			const size_t outIdx = batchHeadOffset + globalRow*headDim + globalCol;
			if(outIdx < totalElements){ Out[outIdx] = __float2half(outAccum[idx]*invSum); }
		}
	}
}
// ============================================================================
// BACKWARD KERNELS
// ============================================================================
__global__ void ComputeDAttDQKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const __half* __restrict__ attention, float* __restrict__ dAtt, __half* __restrict__ dQ, int batchSize, int tokens, int headDim, int heads, int tileCols){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || rowBlock*16 >= tokens) return;
	const int rowStart = rowBlock*16;
	const size_t batchHead = static_cast<size_t>(batch)*heads + head;
	const size_t embOffset = batchHead*tokens*headDim;
	const size_t attOffset = batchHead*tokens*tokens;
	const size_t totalEmbElements = static_cast<size_t>(batchSize)*heads*tokens*headDim;
	const size_t totalAttElements = static_cast<size_t>(batchSize)*heads*tokens*tokens;
	const int numKeyBlocks = (tokens + 15)/16;
	const int numDBlocks = (headDim + 15)/16;
	if(numDBlocks > kMaxValueBlocks || numDBlocks <= 0) return;
	const int warpId = threadIdx.x/32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x/32;
	if(numWarps <= 0 || numWarps > 8) return;
	extern __shared__ char sharedBytes[];
	const int tileStride = 16 + kSharedMemPad;
	const int paddedTileElements = tileStride*16;
	auto scoreTiles = reinterpret_cast<float*>(sharedBytes);
	float* rowSums = scoreTiles + numWarps*paddedTileElements;
	auto outTile = reinterpret_cast<__half*>(rowSums + 16);
	__half* valueTiles = outTile + paddedTileElements;
	__half* attTiles = valueTiles + numWarps*paddedTileElements;
	// Initialize row sums
	if(threadIdx.x < 16){
		const int row = threadIdx.x;
		const int globalRow = rowStart + row;
		if(globalRow < tokens){ rowSums[row] = 0.0f; }
	}
	__syncthreads();
	// Compute dAtt = dOut @ V^T
	for(int tileStart = 0; tileStart < tokens; tileStart += tileCols){
		const int remaining = (tokens - tileStart < tileCols) ? (tokens - tileStart) : tileCols;
		if(remaining <= 0) break;
		for(int colBlock = 0; colBlock < remaining; colBlock += 16*numWarps){
			const int remainingCols = (remaining - colBlock < 16*numWarps) ? (remaining - colBlock) : (16*numWarps);
			const int activeWarps = (remainingCols + 15)/16;
			if(activeWarps <= 0 || activeWarps > numWarps) continue;
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpScores;
			if(warpId < activeWarps){ fill_fragment(warpScores, 0.0f); }
			for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
				if(dBlock*16 >= headDim) break;
				// Load dOut tile for this value block using half2 transactions
				const int colBase = dBlock*16;
				const __half2 zero2 = __float2half2_rn(0.0f);
				constexpr int kVecCols = 8;
				for(int row = threadIdx.x; row < 16; row += blockDim.x){
					__half* rowDst = outTile + row*tileStride;
					auto dstVec = reinterpret_cast<__half2*>(rowDst);
					const int globalRow = rowStart + row;
					if(globalRow >= tokens || colBase >= headDim){
						for(int vec = 0; vec < kVecCols; ++vec){ dstVec[vec] = zero2; }
						continue;
					}
					const size_t rowOffset = embOffset + static_cast<size_t>(globalRow)*headDim;
					if(rowOffset >= totalEmbElements){
						for(int vec = 0; vec < kVecCols; ++vec){ dstVec[vec] = zero2; }
						continue;
					}
					const size_t tileBase = rowOffset + colBase;
					if(tileBase >= totalEmbElements){
						for(int vec = 0; vec < kVecCols; ++vec){ dstVec[vec] = zero2; }
						continue;
					}
					const __half* srcRow = dOut + tileBase;
					for(int vec = 0; vec < kVecCols; ++vec){
						const int globalCol = colBase + vec*2;
						__half2 packed = zero2;
						if(globalCol < headDim){
							const size_t elemIdx = tileBase + static_cast<size_t>(vec)*2;
							if(globalCol + 1 < headDim && elemIdx + 1 < totalEmbElements){
								packed = reinterpret_cast<const __half2*>(srcRow)[vec];
							} else{
								__half lo = __float2half(0.0f);
								__half hi = __float2half(0.0f);
								if(elemIdx < totalEmbElements){ lo = srcRow[vec*2]; }
								if(globalCol + 1 < headDim && elemIdx + 1 < totalEmbElements){
									hi = srcRow[vec*2 + 1];
								}
								packed = __halves2half2(lo, hi);
							}
						}
						dstVec[vec] = packed;
					}
				}
				__syncthreads();
				wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> outFrag;
				if(warpId < activeWarps){ load_matrix_sync(outFrag, outTile, tileStride); }
				// Load V tiles using packed row pairs
				const __half zeroHalf = __float2half(0.0f);
				constexpr int kVecRows = 8;
				const int vecTileCount = activeWarps*16*kVecRows;
				for(int pairIdx = threadIdx.x; pairIdx < vecTileCount; pairIdx += blockDim.x){
					const int warpLocal = pairIdx/(16*kVecRows);
					const int rem = pairIdx % (16*kVecRows);
					const int c = rem/kVecRows;
					const int rowPair = rem % kVecRows;
					const int r0 = rowPair*2;
					const int r1 = r0 + 1;
					const int globalKey = tileStart + colBlock + warpLocal*16 + c;
					const int globalDim0 = dBlock*16 + r0;
					const int globalDim1 = dBlock*16 + r1;
					__half h0 = zeroHalf;
					__half h1 = zeroHalf;
					if(globalKey < tokens){
						if(globalDim0 < headDim && warpLocal*16 + c < remainingCols){
							const size_t vIdx0 = embOffset + static_cast<size_t>(globalKey)*headDim + globalDim0;
							if(vIdx0 < totalEmbElements){ h0 = V[vIdx0]; }
						}
						if(r1 < 16 && globalDim1 < headDim && warpLocal*16 + c < remainingCols){
							const size_t vIdx1 = embOffset + static_cast<size_t>(globalKey)*headDim + globalDim1;
							if(vIdx1 < totalEmbElements){ h1 = V[vIdx1]; }
						}
					}
					__half2 packed = __halves2half2(h0, h1);
					__half* tileBase = valueTiles + warpLocal*paddedTileElements + c*tileStride + r0;
					reinterpret_cast<__half2*>(tileBase)[0] = packed;
				}
				__syncthreads();
				if(warpId < activeWarps){
					wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> vFrag;
					load_matrix_sync(vFrag, valueTiles + warpId*paddedTileElements, tileStride);
					mma_sync(warpScores, outFrag, vFrag, warpScores);
				}
				__syncthreads();
			}
			if(warpId < activeWarps){ store_matrix_sync(scoreTiles + warpId*paddedTileElements, warpScores, tileStride, wmma::mem_row_major); }
			__syncthreads();
			// Store dAtt and compute row sums
#pragma unroll 4
			for(int idx = threadIdx.x; idx < activeWarps*16*16; idx += blockDim.x){
				const int warpLocal = idx/(16*16);
				const int tileIndex = idx % (16*16);
				const int r = tileIndex/16;
				const int c = tileIndex % 16;
				const int globalRow = rowStart + r;
				const int globalCol = tileStart + colBlock + warpLocal*16 + c;
				if(globalRow < tokens && globalCol < tokens && warpLocal*16 + c < remainingCols){
					const size_t attIdx = attOffset + globalRow*tokens + globalCol;
					if(attIdx < totalAttElements){ dAtt[attIdx] = scoreTiles[warpLocal*paddedTileElements + r*tileStride + c]; }
				}
			}
			__syncthreads();
			// Accumulate row sums
#pragma unroll 2
			for(int row = warpId; row < 16; row += numWarps){
				const int globalRow = rowStart + row;
				if(globalRow >= tokens) continue;
				float accum = 0.0f;
				for(int warpLocal = 0; warpLocal < activeWarps; ++warpLocal){
					const int warpColBase = warpLocal*16;
					const int validCols = (remainingCols - warpColBase < 16) ? (remainingCols - warpColBase) : 16;
					if(validCols <= 0) continue;
					const float* tile = scoreTiles + warpLocal*paddedTileElements + row*tileStride;
#pragma unroll 4
					for(int c = laneId; c < validCols; c += 32){
						const int globalCol = tileStart + colBlock + warpColBase + c;
						if(globalCol < tokens){
							const size_t attIdx = attOffset + globalRow*tokens + globalCol;
							if(attIdx < totalAttElements){
								const float rawVal = tile[c];
								const float attVal = __half2float(attention[attIdx]);
								accum += rawVal*attVal;
							}
						}
					}
				}
				accum = WarpReduceSum(accum);
				if(laneId == 0){ atomicAdd(&rowSums[row], accum); }
			}
			__syncthreads();
		}
	}
	__syncthreads();
	// Apply softmax gradient
	for(int tileStart = 0; tileStart < tokens; tileStart += tileCols){
		const int remaining = (tokens - tileStart < tileCols) ? (tokens - tileStart) : tileCols;
		if(remaining <= 0) break;
		for(int colBlock = 0; colBlock < remaining; colBlock += 16*numWarps){
			const int remainingCols = (remaining - colBlock < 16*numWarps) ? (remaining - colBlock) : (16*numWarps);
			if(remainingCols <= 0) continue;
			const int activeWarps = (remainingCols + 15)/16;
#pragma unroll 4
			for(int idx = threadIdx.x; idx < activeWarps*16*16; idx += blockDim.x){
				const int warpLocal = idx/(16*16);
				const int tileIndex = idx % (16*16);
				const int r = tileIndex/16;
				const int c = tileIndex % 16;
				const int globalRow = rowStart + r;
				const int globalCol = tileStart + colBlock + warpLocal*16 + c;
				if(globalRow < tokens && globalCol < tokens && warpLocal*16 + c < remainingCols){
					const size_t attIdx = attOffset + globalRow*tokens + globalCol;
					if(attIdx < totalAttElements){
						const float rawVal = dAtt[attIdx];
						const float attVal = __half2float(attention[attIdx]);
						const float gradVal = attVal*(rawVal - rowSums[r]);
						dAtt[attIdx] = gradVal;
					}
				}
			}
		}
		__syncthreads();
	}
	// Compute dQ = scale*(dAtt @ K)
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		if(dBlock*16 >= headDim) break;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int keyBlock = warpId; keyBlock < numKeyBlocks; keyBlock += numWarps){
			const int keyBase = keyBlock*16;
			if(keyBase >= tokens) continue;
			__half* attTile = attTiles + warpId*paddedTileElements;
			__half* kTile = valueTiles + warpId*paddedTileElements;
			// Load attention tile using half2 stores
			constexpr int kVecCols = 8;
			for(int pairIdx = laneId; pairIdx < 16*kVecCols; pairIdx += 32){
				const int r = pairIdx/kVecCols;
				const int vec = pairIdx % kVecCols;
				const int c0 = vec*2;
				const int c1 = c0 + 1;
				const int globalRow = rowStart + r;
				const int globalCol0 = keyBase + c0;
				const int globalCol1 = keyBase + c1;
				float val0 = 0.0f;
				float val1 = 0.0f;
				if(globalRow < tokens){
					if(globalCol0 < tokens){
						const size_t attIdx0 = attOffset + static_cast<size_t>(globalRow)*tokens + globalCol0;
						if(attIdx0 < totalAttElements){ val0 = dAtt[attIdx0]; }
					}
					if(globalCol1 < tokens){
						const size_t attIdx1 = attOffset + static_cast<size_t>(globalRow)*tokens + globalCol1;
						if(attIdx1 < totalAttElements){ val1 = dAtt[attIdx1]; }
					}
				}
				__half2 packed = __halves2half2(__float2half(val0), __float2half(val1));
				__half* tileBase = attTile + r*tileStride + c0;
				reinterpret_cast<__half2*>(tileBase)[0] = packed;
			}
			// Load K tile with vectorized global reads
			const __half2 zeroPair = __float2half2_rn(0.0f);
			for(int pairIdx = laneId; pairIdx < 16*kVecCols; pairIdx += 32){
				const int r = pairIdx/kVecCols;
				const int vec = pairIdx % kVecCols;
				const int c0 = vec*2;
				const int globalRow = keyBase + r;
				const int globalCol0 = dBlock*16 + c0;
				__half2 packed = zeroPair;
				if(globalRow < tokens && globalCol0 < headDim){
					const size_t baseIdx = embOffset + static_cast<size_t>(globalRow)*headDim + globalCol0;
					if(baseIdx < totalEmbElements){
						if(globalCol0 + 1 < headDim && baseIdx + 1 < totalEmbElements){
							packed = reinterpret_cast<const __half2*>(K + baseIdx)[0];
						} else{
							__half h0 = K[baseIdx];
							__half h1 = __float2half(0.0f);
							if(globalCol0 + 1 < headDim && baseIdx + 1 < totalEmbElements){ h1 = K[baseIdx + 1]; }
							packed = __halves2half2(h0, h1);
						}
					}
				}
				__half* tileBase = kTile + r*tileStride + c0;
				reinterpret_cast<__half2*>(tileBase)[0] = packed;
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> attFrag;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> kFrag;
			load_matrix_sync(attFrag, attTile, tileStride);
			load_matrix_sync(kFrag, kTile, tileStride);
			mma_sync(warpAcc, attFrag, kFrag, warpAcc);
			__syncwarp();
		}
		store_matrix_sync(scoreTiles + warpId*paddedTileElements, warpAcc, tileStride, wmma::mem_row_major);
		__syncthreads();
		// Reduce and store dQ
		ReduceStoreTile16(scoreTiles, tileStride, numWarps, paddedTileElements, dQ, embOffset, rowStart, dBlock*16, tokens, headDim, headDim, totalEmbElements, scale);
		__syncthreads();
	}
}
__global__ void ComputeDVKernel(const __half* __restrict__ attention, const __half* __restrict__ dOut, __half* __restrict__ dV, int batchSize, int tokens, int headDim, int heads){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int keyBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || keyBlock*16 >= tokens) return;
	const int keyStart = keyBlock*16;
	const size_t batchHead = static_cast<size_t>(batch)*heads + head;
	const size_t embOffset = batchHead*tokens*headDim;
	const size_t attOffset = batchHead*tokens*tokens;
	const size_t totalEmbElements = static_cast<size_t>(batchSize)*heads*tokens*headDim;
	const size_t totalAttElements = static_cast<size_t>(batchSize)*heads*tokens*tokens;
	const int numRowBlocks = (tokens + 15)/16;
	const int numDBlocks = (headDim + 15)/16;
	if(numDBlocks > kMaxValueBlocks || numDBlocks <= 0) return;
	const int warpId = threadIdx.x/32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x/32;
	if(numWarps <= 0 || numWarps > 8) return;
	extern __shared__ char sharedBytes[];
	const int tileStride = 16 + kSharedMemPad;
	const int paddedTileElements = tileStride*16;
	auto accumStore = reinterpret_cast<float*>(sharedBytes);
	auto attTiles = reinterpret_cast<__half*>(accumStore + numWarps*paddedTileElements);
	__half* outTiles = attTiles + numWarps*paddedTileElements;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		if(dBlock*16 >= headDim) break;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
			const int queryBase = rowBlock*16;
			if(queryBase >= tokens) continue;
			__half* attTile = attTiles + warpId*paddedTileElements;
			// Load attention tile (transposed) using half2 stores
			constexpr int kVecRows = 8;
			const __half zeroHalf = __float2half(0.0f);
			for(int pairIdx = laneId; pairIdx < 16*kVecRows; pairIdx += 32){
				const int c = pairIdx/kVecRows;
				const int rowPair = pairIdx % kVecRows;
				const int r0 = rowPair*2;
				const int r1 = r0 + 1;
				const int globalKey0 = keyStart + r0;
				const int globalKey1 = keyStart + r1;
				const int globalQuery = queryBase + c;
				__half h0 = zeroHalf;
				__half h1 = zeroHalf;
				if(globalQuery < tokens){
					if(globalKey0 < tokens){
						const size_t attIdx0 = attOffset + static_cast<size_t>(globalQuery)*tokens + globalKey0;
						if(attIdx0 < totalAttElements){ h0 = attention[attIdx0]; }
					}
					if(r1 < 16 && globalKey1 < tokens){
						const size_t attIdx1 = attOffset + static_cast<size_t>(globalQuery)*tokens + globalKey1;
						if(attIdx1 < totalAttElements){ h1 = attention[attIdx1]; }
					}
				}
				__half* tileBase = attTile + c*tileStride + r0;
				reinterpret_cast<__half2*>(tileBase)[0] = __halves2half2(h0, h1);
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			load_matrix_sync(attFrag, attTile, tileStride);
			__half* outTile = outTiles + warpId*paddedTileElements;
			// Load dOut tile using half2 transactions
			const __half2 zeroPair = __float2half2_rn(0.0f);
			for(int pairIdx = laneId; pairIdx < 16*kVecRows; pairIdx += 32){
				const int r = pairIdx/kVecRows;
				const int vec = pairIdx % kVecRows;
				const int c0 = vec*2;
				const int globalQuery = queryBase + r;
				const int globalCol0 = dBlock*16 + c0;
				__half2 packed = zeroPair;
				if(globalQuery < tokens && globalCol0 < headDim){
					const size_t baseIdx = embOffset + static_cast<size_t>(globalQuery)*headDim + globalCol0;
					if(baseIdx < totalEmbElements){
						if(globalCol0 + 1 < headDim && baseIdx + 1 < totalEmbElements){
							packed = reinterpret_cast<const __half2*>(dOut + baseIdx)[0];
						} else{
							__half h0 = dOut[baseIdx];
							__half h1 = __float2half(0.0f);
							if(globalCol0 + 1 < headDim && baseIdx + 1 < totalEmbElements){ h1 = dOut[baseIdx + 1]; }
							packed = __halves2half2(h0, h1);
						}
					}
				}
				__half* tileBase = outTile + r*tileStride + c0;
				reinterpret_cast<__half2*>(tileBase)[0] = packed;
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> outFrag;
			load_matrix_sync(outFrag, outTile, tileStride);
			mma_sync(warpAcc, attFrag, outFrag, warpAcc);
			__syncwarp();
		}
		store_matrix_sync(accumStore + warpId*paddedTileElements, warpAcc, tileStride, wmma::mem_row_major);
		__syncthreads();
		// Reduce and store dV
		ReduceStoreTile16(accumStore, tileStride, numWarps, paddedTileElements, dV, embOffset, keyStart, dBlock*16, tokens, headDim, headDim, totalEmbElements);
		__syncthreads();
	}
}
__global__ void ComputeDKKernel(const float* __restrict__ dAtt, const __half* __restrict__ Q, __half* __restrict__ dK, int batchSize, int tokens, int headDim, int heads){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int keyBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || keyBlock*16 >= tokens) return;
	const int keyStart = keyBlock*16;
	const size_t batchHead = static_cast<size_t>(batch)*heads + head;
	const size_t embOffset = batchHead*tokens*headDim;
	const size_t attOffset = batchHead*tokens*tokens;
	const size_t totalEmbElements = static_cast<size_t>(batchSize)*heads*tokens*headDim;
	const size_t totalAttElements = static_cast<size_t>(batchSize)*heads*tokens*tokens;
	const int numRowBlocks = (tokens + 15)/16;
	const int numDBlocks = (headDim + 15)/16;
	if(numDBlocks > kMaxValueBlocks || numDBlocks <= 0) return;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	const int warpId = threadIdx.x/32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x/32;
	if(numWarps <= 0 || numWarps > 8) return;
	extern __shared__ char sharedBytes[];
	const int tileStride = 16 + kSharedMemPad;
	const int paddedTileElements = tileStride*16;
	auto accumStore = reinterpret_cast<float*>(sharedBytes);
	auto attTiles = reinterpret_cast<__half*>(accumStore + numWarps*paddedTileElements);
	__half* qTiles = attTiles + numWarps*paddedTileElements;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		if(dBlock*16 >= headDim) break;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
			const int queryBase = rowBlock*16;
			if(queryBase >= tokens) continue;
			__half* attTile = attTiles + warpId*paddedTileElements;
			// Load dAtt tile (transposed) using half2 stores
			constexpr int kVecRows = 8;
			const __half zeroHalf = __float2half(0.0f);
			for(int pairIdx = laneId; pairIdx < 16*kVecRows; pairIdx += 32){
				const int c = pairIdx/kVecRows;
				const int rowPair = pairIdx % kVecRows;
				const int r0 = rowPair*2;
				const int r1 = r0 + 1;
				const int globalKey0 = keyStart + r0;
				const int globalKey1 = keyStart + r1;
				const int globalQuery = queryBase + c;
				__half h0 = zeroHalf;
				__half h1 = zeroHalf;
				if(globalQuery < tokens){
					if(globalKey0 < tokens){
						const size_t attIdx0 = attOffset + static_cast<size_t>(globalQuery)*tokens + globalKey0;
						if(attIdx0 < totalAttElements){ h0 = __float2half(dAtt[attIdx0]); }
					}
					if(r1 < 16 && globalKey1 < tokens){
						const size_t attIdx1 = attOffset + static_cast<size_t>(globalQuery)*tokens + globalKey1;
						if(attIdx1 < totalAttElements){ h1 = __float2half(dAtt[attIdx1]); }
					}
				}
				__half* tileBase = attTile + c*tileStride + r0;
				reinterpret_cast<__half2*>(tileBase)[0] = __halves2half2(h0, h1);
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			load_matrix_sync(attFrag, attTile, tileStride);
			__half* qTile = qTiles + warpId*paddedTileElements;
			// Load Q tile using half2 transactions
			const __half2 zeroPair = __float2half2_rn(0.0f);
			for(int pairIdx = laneId; pairIdx < 16*kVecRows; pairIdx += 32){
				const int r = pairIdx/kVecRows;
				const int vec = pairIdx % kVecRows;
				const int c0 = vec*2;
				const int globalQuery = queryBase + r;
				const int globalCol0 = dBlock*16 + c0;
				__half2 packed = zeroPair;
				if(globalQuery < tokens && globalCol0 < headDim){
					const size_t baseIdx = embOffset + static_cast<size_t>(globalQuery)*headDim + globalCol0;
					if(baseIdx < totalEmbElements){
						if(globalCol0 + 1 < headDim && baseIdx + 1 < totalEmbElements){
							packed = reinterpret_cast<const __half2*>(Q + baseIdx)[0];
						} else{
							__half h0 = Q[baseIdx];
							__half h1 = __float2half(0.0f);
							if(globalCol0 + 1 < headDim && baseIdx + 1 < totalEmbElements){ h1 = Q[baseIdx + 1]; }
							packed = __halves2half2(h0, h1);
						}
					}
				}
				__half* tileBase = qTile + r*tileStride + c0;
				reinterpret_cast<__half2*>(tileBase)[0] = packed;
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> qFrag;
			load_matrix_sync(qFrag, qTile, tileStride);
			mma_sync(warpAcc, attFrag, qFrag, warpAcc);
			__syncwarp();
		}
		store_matrix_sync(accumStore + warpId*paddedTileElements, warpAcc, tileStride, wmma::mem_row_major);
		__syncthreads();
		// Reduce and store dK
		ReduceStoreTile16(accumStore, tileStride, numWarps, paddedTileElements, dK, embOffset, keyStart, dBlock*16, tokens, headDim, headDim, totalEmbElements, scale);
		__syncthreads();
	}
}
// ============================================================================
// WRAPPER FUNCTIONS
// ============================================================================
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, __half* AttentionWeights, const float* attentionMask, const float* relPosBias, const int* relPosIndex, int relPosSize, int batchSize, int tokens, int headDim, int heads, int maskBatchSize, int maskHeads){
	size_t sharedMemRequired;
	if(!ValidateAttentionDimensions(batchSize, tokens, headDim, heads, sharedMemRequired)){
		printf("WmmaAttention: Invalid dimensions, aborting\n");
		return;
	}
	if(!Q || !K || !V || !Out){
		printf("WmmaAttention: Null input/output pointer(s)\n");
		return;
	}
	constexpr int threadsPerBlock = kDefaultThreads;
	const int numRowBlocks = DivCeil(tokens, 16);
	if(numRowBlocks > 65535 || batchSize > 65535 || heads > 65535){
		printf("WmmaAttention: Grid dimensions exceed limits (blocks=%d, batch=%d, heads=%d)\n", numRowBlocks, batchSize, heads);
		return;
	}
	dim3 block(threadsPerBlock);
	dim3 grid(numRowBlocks, batchSize, heads);
	const int tileCols = GetAttentionTileCols(tokens);
	cudaError_t err = cudaFuncSetAttribute(WmmaAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSharedMemory);
	if(err != cudaSuccess){
		printf("WmmaAttention: Failed to set shared memory size: %s\n", cudaGetErrorString(err));
		return;
	}
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	WmmaAttentionKernel<<<grid, block, sharedMemRequired>>>(Q, K, V, Out, AttentionWeights, attentionMask, relPosBias, relPosIndex, relPosSize, batchSize, tokens, headDim, heads, tileCols, maskBatchSize, maskHeads);
	checkCUDA(cudaGetLastError());
}
void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const __half* Att, __half* dQ, __half* dK, __half* dV, float* dAttWorkspace, size_t workspaceElements, int batchSize, int tokens, int headDim, int heads){
	if(!Q || !K || !V || !dOut || !Att || !dQ || !dK || !dV || !dAttWorkspace){
		printf("WmmaAttentionBackward: Null pointer(s) provided\n");
		return;
	}
	const size_t requiredElements = static_cast<size_t>(batchSize)*heads*tokens*tokens;
	if(requiredElements > workspaceElements){
		printf("WmmaAttentionBackward: Workspace too small (%zu required, %zu provided)\n", requiredElements, workspaceElements);
		return;
	}
	size_t sharedMemRequired;
	if(!ValidateAttentionDimensions(batchSize, tokens, headDim, heads, sharedMemRequired)){
		printf("WmmaAttentionBackward: Invalid dimensions\n");
		return;
	}
	const int numRowBlocks = DivCeil(tokens, 16);
	const int numKeyBlocks = DivCeil(tokens, 16);
	const int tileCols = GetAttentionTileCols(tokens);
	// Validate grid dimensions
	if(numRowBlocks > 65535 || numKeyBlocks > 65535 || batchSize > 65535 || heads > 65535){
		printf("WmmaAttentionBackward: Grid dimensions exceed limits\n");
		return;
	}
	dim3 block(kDefaultThreads);
	dim3 gridDQ(numRowBlocks, batchSize, heads);
	dim3 gridKV(numKeyBlocks, batchSize, heads);
	const int warpCount = block.x/32;
	const int tileStride = 16 + kSharedMemPad;
	const size_t paddedTileElements = static_cast<size_t>(tileStride)*16;
	const size_t smemDQ = sizeof(float)*(warpCount*paddedTileElements + 16) + sizeof(__half)*((1 + 2*warpCount)*paddedTileElements);
	const size_t smemKV = sizeof(float)*(warpCount*paddedTileElements) + sizeof(__half)*(2*warpCount*paddedTileElements);
	if(smemDQ > kMaxSharedMemory || smemKV > kMaxSharedMemory){
		printf("WmmaAttentionBackward: Shared memory requirements exceed limits\n");
		return;
	}
	// Set shared memory configurations
	cudaFuncSetAttribute(ComputeDAttDQKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSharedMemory);
	cudaFuncSetAttribute(ComputeDVKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSharedMemory);
	cudaFuncSetAttribute(ComputeDKKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSharedMemory);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	// Launch kernels
	ComputeDAttDQKernel<<<gridDQ, block, smemDQ>>>(Q, K, V, dOut, Att, dAttWorkspace, dQ, batchSize, tokens, headDim, heads, tileCols);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttentionBackward dAtt+dQ error: %s\n", cudaGetErrorString(err));
		return;
	}
	ComputeDVKernel<<<gridKV, block, smemKV>>>(Att, dOut, dV, batchSize, tokens, headDim, heads);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttentionBackward dV error: %s\n", cudaGetErrorString(err));
		return;
	}
	ComputeDKKernel<<<gridKV, block, smemKV>>>(dAttWorkspace, Q, dK, batchSize, tokens, headDim, heads);
	err = cudaGetLastError();
	if(err != cudaSuccess){ printf("WmmaAttentionBackward dK error: %s\n", cudaGetErrorString(err)); }
}

__global__ void RelPosBiasGradKernel(const float* __restrict__ dAtt, const int* __restrict__ relPosIndex, float* __restrict__ gradBias, int batchSize, int tokens, int heads, int relPosSize, float scale){
	const size_t total = static_cast<size_t>(batchSize)*heads*tokens*tokens;
	const size_t idx = static_cast<size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
	if(idx >= total) return;
	const int col = static_cast<int>(idx % tokens);
	const int row = static_cast<int>((idx / tokens) % tokens);
	const int head = static_cast<int>((idx / (static_cast<size_t>(tokens)*tokens)) % heads);
	const int relIdx = relPosIndex[row*tokens + col];
	const size_t biasIdx = static_cast<size_t>(head)*relPosSize + relIdx;
	atomicAdd(&gradBias[biasIdx], dAtt[idx]*scale);
}

void AccumulateRelPosBiasGrad(const float* dAtt, const int* relPosIndex, float* gradBias, int batchSize, int tokens, int heads, int relPosSize, float scale){
	if(dAtt == nullptr || relPosIndex == nullptr || gradBias == nullptr) return;
	const size_t total = static_cast<size_t>(batchSize)*heads*tokens*tokens;
	if(total == 0 || relPosSize <= 0) return;
	size_t blocks, tpb = 256;
	GetLaunchConfigGridStride(total, blocks, tpb);
	RelPosBiasGradKernel<<<blocks, tpb>>>(dAtt, relPosIndex, gradBias, batchSize, tokens, heads, relPosSize, scale);
	checkCUDA(cudaGetLastError());
}
// ============================================================================
// PACKING/UNPACKING UTILITIES
// ============================================================================
__global__ void PackColumnsToHeadsKernel(const __half* __restrict__ inputQ, const __half* __restrict__ inputK,
	const __half* __restrict__ inputV,
	__half* __restrict__ outputQ, __half* __restrict__ outputK,
	__half* __restrict__ outputV,
	int B, int T, int H, int D){
	const int total = B*T*H*D;
	const int embedDim = H*D;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += blockDim.x*gridDim.x){
		const int d = idx % D;
		int tmp = idx/D;
		const int t = tmp % T;
		tmp /= T;
		const int h = tmp % H;
		const int b = tmp/H;
		const int col = b*T + t;
		const int row = h*D + d;
		const size_t inIdx = static_cast<size_t>(row) + static_cast<size_t>(col)*embedDim;
		if(inputQ && outputQ){ outputQ[idx] = inputQ[inIdx]; }
		if(inputK && outputK){ outputK[idx] = inputK[inIdx]; }
		if(inputV && outputV){ outputV[idx] = inputV[inIdx]; }
	}
}
static void LaunchPackColumnsToHeadsKernel(const __half* inputQ, const __half* inputK, const __half* inputV,
	__half* outputQ, __half* outputK, __half* outputV,
	int batch, int tokens, int embedDim, int numHeads){
	if((inputQ && !outputQ) || (inputK && !outputK) || (inputV && !outputV) || (!inputQ && outputQ) || (!inputK && outputK) || (!inputV && outputV)){
		printf("PackColumnsToHeads: Mismatched input/output pointers\n");
		return;
	}
	if(!inputQ && !inputK && !inputV){
		printf("PackColumnsToHeads: No input tensors provided\n");
		return;
	}
	if(numHeads <= 0 || numHeads > 128){
		printf("PackColumnsToHeads: Invalid head count %d\n", numHeads);
		return;
	}
	if(embedDim % numHeads != 0){
		printf("PackColumnsToHeads: embedDim %d not divisible by numHeads %d\n", embedDim, numHeads);
		return;
	}
	if(batch <= 0 || batch > kMaxBatch || tokens <= 0 || tokens > kMaxTokens){
		printf("PackColumnsToHeads: Invalid dimensions B=%d, T=%d\n", batch, tokens);
		return;
	}
	const int headDim = embedDim/numHeads;
	const int total = batch*tokens*embedDim;
	if(total <= 0){
		return;
	}
	constexpr int blockSize = 256;
	const int gridSize = std::min(65535, std::max(1, DivCeil(total, blockSize)));
	PackColumnsToHeadsKernel<<<gridSize, blockSize>>>(inputQ, inputK, inputV, outputQ, outputK, outputV, batch, tokens, numHeads, headDim);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){ printf("PackColumnsToHeads error: %s\n", cudaGetErrorString(err)); }
}
void PackColumnsToHeads(const __half* inputQ, const __half* inputK, const __half* inputV,
	__half* outputQ, __half* outputK, __half* outputV,
	int batch, int tokens, int embedDim, int numHeads){
	LaunchPackColumnsToHeadsKernel(inputQ, inputK, inputV, outputQ, outputK, outputV, batch, tokens, embedDim, numHeads);
}
void PackColumnsToHeads(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads){
	if(!input || !output){
		printf("PackColumnsToHeads: Null pointer(s)\n");
		return;
	}
	LaunchPackColumnsToHeadsKernel(input, nullptr, nullptr, output, nullptr, nullptr, batch, tokens, embedDim, numHeads);
}
__global__ void PackHeadsToColumnsKernel(const __half* __restrict__ inputQ, const __half* __restrict__ inputK,
	const __half* __restrict__ inputV,
	__half* __restrict__ outputQ, __half* __restrict__ outputK,
	__half* __restrict__ outputV,
	int B, int T, int H, int D){
	const int total = B*T*H*D;
	const int embedDim = H*D;
	for(int idx = blockIdx.x*blockDim.x + threadIdx.x; idx < total; idx += blockDim.x*gridDim.x){
		const int d = idx % D;
		int tmp = idx/D;
		const int t = tmp % T;
		tmp /= T;
		const int h = tmp % H;
		const int b = tmp/H;
		const int col = b*T + t;
		const int row = h*D + d;
		const size_t outIdx = static_cast<size_t>(row) + static_cast<size_t>(col)*embedDim;
		if(inputQ && outputQ){ outputQ[outIdx] = inputQ[idx]; }
		if(inputK && outputK){ outputK[outIdx] = inputK[idx]; }
		if(inputV && outputV){ outputV[outIdx] = inputV[idx]; }
	}
}
static void LaunchPackHeadsToColumnsKernel(const __half* inputQ, const __half* inputK, const __half* inputV,
	__half* outputQ, __half* outputK, __half* outputV,
	int batch, int tokens, int embedDim, int numHeads){
	if((inputQ && !outputQ) || (inputK && !outputK) || (inputV && !outputV) || (!inputQ && outputQ) || (!inputK && outputK) || (!inputV && outputV)){
		printf("PackHeadsToColumns: Mismatched input/output pointers\n");
		return;
	}
	if(!inputQ && !inputK && !inputV){
		printf("PackHeadsToColumns: No input tensors provided\n");
		return;
	}
	if(numHeads <= 0 || numHeads > 128){
		printf("PackHeadsToColumns: Invalid head count %d\n", numHeads);
		return;
	}
	if(embedDim % numHeads != 0){
		printf("PackHeadsToColumns: embedDim %d not divisible by numHeads %d\n", embedDim, numHeads);
		return;
	}
	if(batch <= 0 || batch > kMaxBatch || tokens <= 0 || tokens > kMaxTokens){
		printf("PackHeadsToColumns: Invalid dimensions B=%d, T=%d\n", batch, tokens);
		return;
	}
	const int headDim = embedDim/numHeads;
	const int total = batch*tokens*embedDim;
	if(total <= 0){
		return;
	}
	constexpr int blockSize = 256;
	const int gridSize = std::min(65535, std::max(1, DivCeil(total, blockSize)));
	PackHeadsToColumnsKernel<<<gridSize, blockSize>>>(inputQ, inputK, inputV, outputQ, outputK, outputV, batch, tokens, numHeads, headDim);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){ printf("PackHeadsToColumns error: %s\n", cudaGetErrorString(err)); }
}
void PackHeadsToColumns(const __half* inputQ, const __half* inputK, const __half* inputV,
	__half* outputQ, __half* outputK, __half* outputV,
	int batch, int tokens, int embedDim, int numHeads){
	LaunchPackHeadsToColumnsKernel(inputQ, inputK, inputV, outputQ, outputK, outputV, batch, tokens, embedDim, numHeads);
}
void PackHeadsToColumns(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads){
	if(!input || !output){
		printf("PackHeadsToColumns: Null pointer(s)\n");
		return;
	}
	LaunchPackHeadsToColumnsKernel(input, nullptr, nullptr, output, nullptr, nullptr, batch, tokens, embedDim, numHeads);
}
