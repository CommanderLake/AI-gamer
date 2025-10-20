#define __CUDACC__
#include "CuCommon.cuh"
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda;
namespace{
	// Configuration constants with safer defaults
	constexpr int kMaxTileCols = 128;
	constexpr int kMaxValueBlocks = 32;
	constexpr int kSharedMemPad = 8;
	constexpr float SOFTMAX_FTZ_THRESHOLD = -12.0f;
	constexpr float SOFTMAX_MAX_INPUT = 20.0f;
	constexpr size_t kMaxSharedMemory = 98304; // 96KB typical max
	constexpr int kMinTokens = 1;
	constexpr int kMaxTokens = 8192;
	constexpr int kMaxHeadDim = 512;
	constexpr int kWarpSize = 32;
	constexpr int kTileSize = 16;
	constexpr int kDefaultThreads = 256;
	// Helper for ceiling division
	__host__ __device__ inline int DivCeil(int a, int b){ return (a + b - 1) / b; }
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
		if(batchSize <= 0 || batchSize > 1024){
			printf("Invalid batch size: %d (must be 1-1024)\n", batchSize);
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
		const int qBlocks = (headDim + 15) / 16;
		if(qBlocks > kMaxValueBlocks){
			printf("Head dimension %d requires %d blocks, exceeds limit %d\n", headDim, qBlocks, kMaxValueBlocks);
			return false;
		}
		// Calculate shared memory requirement
		const int tileCols = GetAttentionTileCols(tokens);
		const int warpCount = kDefaultThreads / 32;
		const int qStride = (headDim + 15) / 16 * 16 + kSharedMemPad;
		const int tileStride = 16 + kSharedMemPad;
		sharedMemRequired = sizeof(__half) * (16 * qStride + warpCount * tileStride * 16 + tileStride * 16) + sizeof(float) * (16 * tileCols + 32);
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
	// Safe memory access helper
	__device__ __forceinline__ bool IsValidIndex(int idx, int max){ return idx >= 0 && idx < max; }
	__device__ __forceinline__ __half SafeLoad(const __half* ptr, int idx, int maxIdx){ return (idx >= 0 && idx < maxIdx) ? ptr[idx] : __float2half(0.0f); }
	__device__ __forceinline__ void SafeStore(__half* ptr, int idx, int maxIdx, __half value){ if(idx >= 0 && idx < maxIdx) ptr[idx] = value; }
	__device__ __forceinline__ void SafeStoreFloat(float* ptr, int idx, int maxIdx, float value){ if(idx >= 0 && idx < maxIdx) ptr[idx] = value; }
}
// ============================================================================
// FORWARD KERNEL
// ============================================================================
__global__ void WmmaAttentionKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ Out, float* __restrict__ AttentionWeights, int batchSize, int tokens, int headDim, int heads, int tileCols){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	// Early exit for out-of-bounds blocks
	if(batch >= batchSize || head >= heads || rowBlock * 16 >= tokens) return;
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	// Validate warp configuration
	if(numWarps == 0 || numWarps > 8) return;
	const size_t batchHeadOffset = (static_cast<size_t>(batch) * heads + head) * tokens * headDim;
	const size_t totalElements = static_cast<size_t>(batchSize) * heads * tokens * headDim;
	extern __shared__ char sharedMemBytes[];
	// Calculate strides with alignment
	const int qStride = ((headDim + 15) / 16 * 16) + kSharedMemPad;
	const int tileStride = 16 + kSharedMemPad;
	auto qShared = reinterpret_cast<__half*>(sharedMemBytes);
	__half* warpTiles = qShared + 16 * qStride;
	__half* attTile = warpTiles + numWarps * tileStride * 16;
	auto scoresTile = reinterpret_cast<float*>(attTile + tileStride * 16);
	float* rowMax = scoresTile + 16 * tileCols;
	float* rowSum = rowMax + 16;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	const __half scaleHalf = __float2half(scale);
	const int qBlocks = (headDim + 15) / 16;
	// Runtime validation
	if(qBlocks > kMaxValueBlocks || qBlocks <= 0) return;
	// WMMA fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frags[kMaxValueBlocks];
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
	// Initialize row statistics
	if(threadIdx.x < 16){
		const int row = threadIdx.x;
		const int globalRow = rowBlock * 16 + row;
		if(globalRow < tokens){
			rowMax[row] = -1e20f; // Use large but not infinite value
			rowSum[row] = 0.0f;
		}
	}
	__syncthreads();
	// Load and scale Q matrix
	for(int row = 0; row < 16; ++row){
		const int globalRow = rowBlock * 16 + row;
		__half* sharedRow = qShared + row * qStride;
		// Initialize shared memory
		for(int col = threadIdx.x; col < qStride; col += blockDim.x){ sharedRow[col] = __float2half(0.0f); }
		__syncthreads();
		if(globalRow < tokens){
			const size_t qRowOffset = batchHeadOffset + globalRow * headDim;
			// Coalesced loading with bounds checking
			for(int col = threadIdx.x; col < headDim; col += blockDim.x){
				const size_t idx = qRowOffset + col;
				if(idx < totalElements){ sharedRow[col] = __hmul(Q[idx], scaleHalf); }
			}
		}
	}
	__syncthreads();
	// Load Q fragments
#pragma unroll
	for(int dBlock = 0; dBlock < qBlocks; ++dBlock){ if(dBlock * 16 < headDim){ load_matrix_sync(q_frags[dBlock], qShared + dBlock * 16, qStride); } }
	const size_t attentionOffset = (static_cast<size_t>(batch) * heads + head) * tokens * tokens;
	const size_t maxAttentionIdx = static_cast<size_t>(batchSize) * heads * tokens * tokens;
	// Process K tiles - compute QK^T
	for(int tileStart = 0; tileStart < tokens; tileStart += tileCols){
		const int remaining = (tokens - tileStart < tileCols) ? (tokens - tileStart) : tileCols;
		if(remaining <= 0) break;
		// Process columns in blocks of 16*numWarps
		for(int colBlock = 0; colBlock < remaining; colBlock += 16 * numWarps){
			const int remainingCols = (remaining - colBlock < 16 * numWarps) ? (remaining - colBlock) : (16 * numWarps);
			const int activeWarps = (remainingCols + 15) / 16;
			if(activeWarps <= 0 || activeWarps > numWarps) continue;
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpScores;
			if(warpId < activeWarps){ fill_fragment(warpScores, 0.0f); }
			// Process K blocks
#pragma unroll 2
			for(int kBlock = 0; kBlock < qBlocks; kBlock++){
				if(kBlock * 16 >= headDim) break;
				// Collaborative K tile loading
				const int elementsPerTile = 16 * 16;
				const int totalElements = elementsPerTile * activeWarps;
#pragma unroll 4
				for(int idx = threadIdx.x; idx < totalElements; idx += blockDim.x){
					const int warpLocal = idx / elementsPerTile;
					const int tileIndex = idx % elementsPerTile;
					const int col = tileIndex / 16;
					const int row = tileIndex % 16;
					const int localCol = warpLocal * 16 + col;
					const int globalCol = tileStart + colBlock + localCol;
					const int globalRow = kBlock * 16 + row;
					__half val = __float2half(0.0f);
					if(globalCol < tokens && globalRow < headDim){
						const size_t kIdx = batchHeadOffset + globalCol * headDim + globalRow;
						if(kIdx < totalElements){ val = K[kIdx]; }
					}
					const int tileIdx = warpLocal * tileStride * 16 + col * tileStride + row;
					warpTiles[tileIdx] = val;
				}
				__syncthreads();
				if(warpId < activeWarps){
					load_matrix_sync(k_frag, warpTiles + warpId * tileStride * 16, tileStride);
					mma_sync(warpScores, q_frags[kBlock], k_frag, warpScores);
				}
				__syncthreads();
			}
			if(warpId < activeWarps){ store_matrix_sync(scoresTile + colBlock + warpId * 16, warpScores, tileCols, wmma::mem_row_major); }
		}
		__syncthreads();
		// Compute softmax statistics
#pragma unroll 4
		for(int row = warpId; row < 16; row += numWarps){
			const int globalRow = rowBlock * 16 + row;
			if(globalRow >= tokens) continue;
			const float prevMax = rowMax[row];
			const float prevSum = rowSum[row];
			// Find max
			float localMax = -1e20f;
#pragma unroll 4
			for(int col = laneId; col < remaining; col += 32){
				const int globalCol = tileStart + col;
				if(globalCol < tokens){
					const float val = scoresTile[row * tileCols + col];
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
					const float val = scoresTile[row * tileCols + col];
					const float diff = val - newMax;
					if(diff > SOFTMAX_FTZ_THRESHOLD && diff < SOFTMAX_MAX_INPUT){ localSum += expf(diff); }
				}
			}
			localSum = WarpReduceSum(localSum);
			if(laneId == 0){
				rowMax[row] = newMax;
				rowSum[row] = prevSum * scalePrev + localSum;
			}
		}
		__syncthreads();
	}
	// Initialize output accumulators
	const int valueBlocks = (headDim + 15) / 16;
	if(valueBlocks > kMaxValueBlocks || valueBlocks <= 0) return;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frags[kMaxValueBlocks];
#pragma unroll
	for(int vb = 0; vb < valueBlocks; ++vb){ fill_fragment(out_frags[vb], 0.0f); }
	// Process V tiles - compute softmax(QK^T)V
	for(int tileStart = 0; tileStart < tokens; tileStart += tileCols){
		const int remaining = (tokens - tileStart < tileCols) ? (tokens - tileStart) : tileCols;
		if(remaining <= 0) break;
		// Apply softmax normalization
#pragma unroll 4
		for(int row = warpId; row < 16; row += numWarps){
			const int globalRow = rowBlock * 16 + row;
			if(globalRow >= tokens) continue;
			const float maxVal = rowMax[row];
			const float sumVal = rowSum[row];
			const float invSum = (sumVal > 1e-10f) ? (1.0f / sumVal) : 0.0f;
#pragma unroll 4
			for(int col = laneId; col < remaining; col += 32){
				const int globalCol = tileStart + col;
				if(globalCol < tokens){
					const float logit = scoresTile[row * tileCols + col];
					const float diff = logit - maxVal;
					float normalized = 0.0f;
					if(diff > SOFTMAX_FTZ_THRESHOLD && diff < SOFTMAX_MAX_INPUT && isfinite(invSum)){
						normalized = expf(diff) * invSum;
						// Clamp to valid range
						normalized = fminf(fmaxf(normalized, 0.0f), 1.0f);
					}
					scoresTile[row * tileCols + col] = normalized;
					// Store attention weights if requested
					if(AttentionWeights != nullptr){
						const size_t attIdx = attentionOffset + globalRow * tokens + globalCol;
						if(attIdx < maxAttentionIdx){ AttentionWeights[attIdx] = normalized; }
					}
				}
			}
		}
		__syncthreads();
		// Matrix multiply with V
		for(int colBlock = 0; colBlock < remaining; colBlock += 16){
			if(colBlock >= remaining) break;
			// Load attention tile
#pragma unroll 4
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int row = idx / 16;
				const int col = idx % 16;
				const int localCol = colBlock + col;
				const int globalCol = tileStart + localCol;
				float val = 0.0f;
				if(rowBlock * 16 + row < tokens && localCol < remaining && globalCol < tokens){ val = scoresTile[row * tileCols + localCol]; }
				attTile[row * tileStride + col] = __float2half(val);
			}
			__syncthreads();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> att_frag;
			load_matrix_sync(att_frag, attTile, tileStride);
			// Process V blocks
			for(int vb = 0; vb < valueBlocks; ++vb){
				if(vb * 16 >= headDim) break;
				// Load V tile
#pragma unroll 4
				for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
					const int row = idx / 16;
					const int col = idx % 16;
					const int keyIdx = tileStart + colBlock + row;
					const int valueIdx = vb * 16 + col;
					__half val = __float2half(0.0f);
					if(keyIdx < tokens && valueIdx < headDim){
						const size_t vIdx = batchHeadOffset + keyIdx * headDim + valueIdx;
						if(vIdx < totalElements){ val = V[vIdx]; }
					}
					warpTiles[row * tileStride + col] = val;
				}
				__syncthreads();
				load_matrix_sync(v_frag, warpTiles, tileStride);
				mma_sync(out_frags[vb], att_frag, v_frag, out_frags[vb]);
				__syncthreads();
			}
		}
	}
	// Store output
	for(int vb = 0; vb < valueBlocks; ++vb){
		if(vb * 16 >= headDim) break;
#pragma unroll
		for(int i = 0; i < out_frags[vb].num_elements; i++){
			const int row = i / 16;
			const int col = i % 16;
			const int globalRow = rowBlock * 16 + row;
			const int globalCol = vb * 16 + col;
			if(globalRow < tokens && globalCol < headDim){
				const size_t outIdx = batchHeadOffset + globalRow * headDim + globalCol;
				if(outIdx < totalElements){ Out[outIdx] = __float2half(out_frags[vb].x[i]); }
			}
		}
	}
}
// ============================================================================
// BACKWARD KERNELS
// ============================================================================
__global__ void ComputeDAttDQKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ attention, float* __restrict__ dAtt, __half* __restrict__ dQ, int batchSize, int tokens, int headDim, int heads,
									int tileCols){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || rowBlock * 16 >= tokens) return;
	const int rowStart = rowBlock * 16;
	const size_t batchHead = static_cast<size_t>(batch) * heads + head;
	const size_t embOffset = batchHead * tokens * headDim;
	const size_t attOffset = batchHead * tokens * tokens;
	const size_t totalEmbElements = static_cast<size_t>(batchSize) * heads * tokens * headDim;
	const size_t totalAttElements = static_cast<size_t>(batchSize) * heads * tokens * tokens;
	const int numKeyBlocks = (tokens + 15) / 16;
	const int numDBlocks = (headDim + 15) / 16;
	if(numDBlocks > kMaxValueBlocks || numDBlocks <= 0) return;
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	if(numWarps <= 0 || numWarps > 8) return;
	extern __shared__ char sharedBytes[];
	const int tileStride = 16 + kSharedMemPad;
	const int paddedTileElements = tileStride * 16;
	auto scoreTiles = reinterpret_cast<float*>(sharedBytes);
	float* rowSums = scoreTiles + numWarps * paddedTileElements;
	auto outTiles = reinterpret_cast<__half*>(rowSums + 16);
	__half* valueTiles = outTiles + kMaxValueBlocks * paddedTileElements;
	__half* attTiles = valueTiles + numWarps * paddedTileElements;
	// Initialize row sums
	if(threadIdx.x < 16){
		const int row = threadIdx.x;
		const int globalRow = rowStart + row;
		if(globalRow < tokens){ rowSums[row] = 0.0f; }
	}
	__syncthreads();
	// Load dOut tiles
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		if(dBlock * 16 >= headDim) break;
#pragma unroll 4
		for(int idx = threadIdx.x; idx < paddedTileElements; idx += blockDim.x){
			const int row = idx / tileStride;
			const int col = idx % tileStride;
			if(row < 16 && col < 16){
				const int globalRow = rowStart + row;
				const int globalCol = dBlock * 16 + col;
				__half val = __float2half(0.0f);
				if(globalRow < tokens && globalCol < headDim){
					const size_t dOutIdx = embOffset + globalRow * headDim + globalCol;
					if(dOutIdx < totalEmbElements){ val = dOut[dOutIdx]; }
				}
				outTiles[dBlock * paddedTileElements + row * tileStride + col] = val;
			}
		}
	}
	__syncthreads();
	// Compute dAtt = dOut @ V^T
	for(int tileStart = 0; tileStart < tokens; tileStart += tileCols){
		const int remaining = (tokens - tileStart < tileCols) ? (tokens - tileStart) : tileCols;
		if(remaining <= 0) break;
		for(int colBlock = 0; colBlock < remaining; colBlock += 16 * numWarps){
			const int remainingCols = (remaining - colBlock < 16 * numWarps) ? (remaining - colBlock) : (16 * numWarps);
			const int activeWarps = (remainingCols + 15) / 16;
			if(activeWarps <= 0 || activeWarps > numWarps) continue;
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpScores;
			if(warpId < activeWarps){ fill_fragment(warpScores, 0.0f); }
			for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
				if(dBlock * 16 >= headDim) break;
				const __half* outTile = outTiles + dBlock * paddedTileElements;
				wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> outFrag;
				load_matrix_sync(outFrag, outTile, tileStride);
				// Load V tiles
#pragma unroll 4
				for(int idx = threadIdx.x; idx < activeWarps * 16 * 16; idx += blockDim.x){
					const int warpLocal = idx / (16 * 16);
					const int tileIndex = idx % (16 * 16);
					const int r = tileIndex / 16;
					const int c = tileIndex % 16;
					const int globalRow = tileStart + colBlock + warpLocal * 16 + r;
					const int globalCol = dBlock * 16 + c;
					__half val = __float2half(0.0f);
					if(globalRow < tokens && globalCol < headDim && warpLocal * 16 + r < remainingCols){
						const size_t vIdx = embOffset + globalRow * headDim + globalCol;
						if(vIdx < totalEmbElements){ val = V[vIdx]; }
					}
					valueTiles[warpLocal * paddedTileElements + c * tileStride + r] = val;
				}
				__syncthreads();
				if(warpId < activeWarps){
					wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> vFrag;
					load_matrix_sync(vFrag, valueTiles + warpId * paddedTileElements, tileStride);
					mma_sync(warpScores, outFrag, vFrag, warpScores);
				}
				__syncthreads();
			}
			if(warpId < activeWarps){ store_matrix_sync(scoreTiles + warpId * paddedTileElements, warpScores, tileStride, wmma::mem_row_major); }
			__syncthreads();
			// Store dAtt and compute row sums
#pragma unroll 4
			for(int idx = threadIdx.x; idx < activeWarps * 16 * 16; idx += blockDim.x){
				const int warpLocal = idx / (16 * 16);
				const int tileIndex = idx % (16 * 16);
				const int r = tileIndex / 16;
				const int c = tileIndex % 16;
				const int globalRow = rowStart + r;
				const int globalCol = tileStart + colBlock + warpLocal * 16 + c;
				if(globalRow < tokens && globalCol < tokens && warpLocal * 16 + c < remainingCols){
					const size_t attIdx = attOffset + globalRow * tokens + globalCol;
					if(attIdx < totalAttElements){ dAtt[attIdx] = scoreTiles[warpLocal * paddedTileElements + r * tileStride + c]; }
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
					const int warpColBase = warpLocal * 16;
					const int validCols = (remainingCols - warpColBase < 16) ? (remainingCols - warpColBase) : 16;
					if(validCols <= 0) continue;
					const float* tile = scoreTiles + warpLocal * paddedTileElements + row * tileStride;
#pragma unroll 4
					for(int c = laneId; c < validCols; c += 32){
						const int globalCol = tileStart + colBlock + warpColBase + c;
						if(globalCol < tokens){
							const size_t attIdx = attOffset + globalRow * tokens + globalCol;
							if(attIdx < totalAttElements){
								const float rawVal = tile[c];
								const float attVal = attention[attIdx];
								accum += rawVal * attVal;
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
		for(int colBlock = 0; colBlock < remaining; colBlock += 16 * numWarps){
			const int remainingCols = (remaining - colBlock < 16 * numWarps) ? (remaining - colBlock) : (16 * numWarps);
			if(remainingCols <= 0) continue;
			const int activeWarps = (remainingCols + 15) / 16;
#pragma unroll 4
			for(int idx = threadIdx.x; idx < activeWarps * 16 * 16; idx += blockDim.x){
				const int warpLocal = idx / (16 * 16);
				const int tileIndex = idx % (16 * 16);
				const int r = tileIndex / 16;
				const int c = tileIndex % 16;
				const int globalRow = rowStart + r;
				const int globalCol = tileStart + colBlock + warpLocal * 16 + c;
				if(globalRow < tokens && globalCol < tokens && warpLocal * 16 + c < remainingCols){
					const size_t attIdx = attOffset + globalRow * tokens + globalCol;
					if(attIdx < totalAttElements){
						const float rawVal = dAtt[attIdx];
						const float attVal = attention[attIdx];
						const float gradVal = attVal * (rawVal - rowSums[r]);
						dAtt[attIdx] = gradVal;
					}
				}
			}
		}
		__syncthreads();
	}
	// Compute dQ = scale * (dAtt @ K)
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		if(dBlock * 16 >= headDim) break;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int keyBlock = warpId; keyBlock < numKeyBlocks; keyBlock += numWarps){
			const int keyBase = keyBlock * 16;
			if(keyBase >= tokens) continue;
			__half* attTile = attTiles + warpId * paddedTileElements;
			__half* kTile = valueTiles + warpId * paddedTileElements;
			// Load attention tile
#pragma unroll 4
			for(int idx = laneId; idx < 16 * 16; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = rowStart + r;
				const int globalCol = keyBase + c;
				float val = 0.0f;
				if(globalRow < tokens && globalCol < tokens){
					const size_t attIdx = attOffset + globalRow * tokens + globalCol;
					if(attIdx < totalAttElements){ val = dAtt[attIdx]; }
				}
				attTile[r * tileStride + c] = __float2half(val);
			}
			// Load K tile
#pragma unroll 4
			for(int idx = laneId; idx < 16 * 16; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = keyBase + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalRow < tokens && globalCol < headDim){
					const size_t kIdx = embOffset + globalRow * headDim + globalCol;
					if(kIdx < totalEmbElements){ val = K[kIdx]; }
				}
				kTile[r * tileStride + c] = val;
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> attFrag;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> kFrag;
			load_matrix_sync(attFrag, attTile, tileStride);
			load_matrix_sync(kFrag, kTile, tileStride);
			mma_sync(warpAcc, attFrag, kFrag, warpAcc);
			__syncwarp();
		}
		store_matrix_sync(scoreTiles + warpId * paddedTileElements, warpAcc, tileStride, wmma::mem_row_major);
		__syncthreads();
		// Reduce and store dQ
#pragma unroll 4
		for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			float sum = 0.0f;
#pragma unroll
			for(int w = 0; w < numWarps; ++w){ sum += scoreTiles[w * paddedTileElements + r * tileStride + c]; }
			const int globalRow = rowStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < tokens && globalCol < headDim){
				const size_t dqIdx = embOffset + globalRow * headDim + globalCol;
				if(dqIdx < totalEmbElements){ dQ[dqIdx] = __float2half(sum * scale); }
			}
		}
		__syncthreads();
	}
}
__global__ void ComputeDVKernel(const float* __restrict__ attention, const __half* __restrict__ dOut, __half* __restrict__ dV, int batchSize, int tokens, int headDim, int heads){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int keyBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || keyBlock * 16 >= tokens) return;
	const int keyStart = keyBlock * 16;
	const size_t batchHead = static_cast<size_t>(batch) * heads + head;
	const size_t embOffset = batchHead * tokens * headDim;
	const size_t attOffset = batchHead * tokens * tokens;
	const size_t totalEmbElements = static_cast<size_t>(batchSize) * heads * tokens * headDim;
	const size_t totalAttElements = static_cast<size_t>(batchSize) * heads * tokens * tokens;
	const int numRowBlocks = (tokens + 15) / 16;
	const int numDBlocks = (headDim + 15) / 16;
	if(numDBlocks > kMaxValueBlocks || numDBlocks <= 0) return;
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	if(numWarps <= 0 || numWarps > 8) return;
	extern __shared__ char sharedBytes[];
	const int tileStride = 16 + kSharedMemPad;
	const int paddedTileElements = tileStride * 16;
	auto accumStore = reinterpret_cast<float*>(sharedBytes);
	auto attTiles = reinterpret_cast<__half*>(accumStore + numWarps * paddedTileElements);
	__half* outTiles = attTiles + numWarps * paddedTileElements;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		if(dBlock * 16 >= headDim) break;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
			const int queryBase = rowBlock * 16;
			if(queryBase >= tokens) continue;
			__half* attTile = attTiles + warpId * paddedTileElements;
			// Load attention tile (transposed)
#pragma unroll 4
			for(int idx = laneId; idx < 16 * 16; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = queryBase + r;
				const int globalKey = keyStart + c;
				float val = 0.0f;
				if(globalQuery < tokens && globalKey < tokens){
					const size_t attIdx = attOffset + globalQuery * tokens + globalKey;
					if(attIdx < totalAttElements){ val = attention[attIdx]; }
				}
				attTile[c * tileStride + r] = __float2half(val);
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			load_matrix_sync(attFrag, attTile, tileStride);
			__half* outTile = outTiles + warpId * paddedTileElements;
			// Load dOut tile
#pragma unroll 4
			for(int idx = laneId; idx < 16 * 16; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = queryBase + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalQuery < tokens && globalCol < headDim){
					const size_t outIdx = embOffset + globalQuery * headDim + globalCol;
					if(outIdx < totalEmbElements){ val = dOut[outIdx]; }
				}
				outTile[r * tileStride + c] = val;
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> outFrag;
			load_matrix_sync(outFrag, outTile, tileStride);
			mma_sync(warpAcc, attFrag, outFrag, warpAcc);
			__syncwarp();
		}
		store_matrix_sync(accumStore + warpId * paddedTileElements, warpAcc, tileStride, wmma::mem_row_major);
		__syncthreads();
		// Reduce and store dV
#pragma unroll 4
		for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			float sum = 0.0f;
#pragma unroll
			for(int w = 0; w < numWarps; ++w){ sum += accumStore[w * paddedTileElements + r * tileStride + c]; }
			const int globalRow = keyStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < tokens && globalCol < headDim){
				const size_t dvIdx = embOffset + globalRow * headDim + globalCol;
				if(dvIdx < totalEmbElements){ dV[dvIdx] = __float2half(sum); }
			}
		}
		__syncthreads();
	}
}
__global__ void ComputeDKKernel(const float* __restrict__ dAtt, const __half* __restrict__ Q, __half* __restrict__ dK, int batchSize, int tokens, int headDim, int heads){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int keyBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || keyBlock * 16 >= tokens) return;
	const int keyStart = keyBlock * 16;
	const size_t batchHead = static_cast<size_t>(batch) * heads + head;
	const size_t embOffset = batchHead * tokens * headDim;
	const size_t attOffset = batchHead * tokens * tokens;
	const size_t totalEmbElements = static_cast<size_t>(batchSize) * heads * tokens * headDim;
	const size_t totalAttElements = static_cast<size_t>(batchSize) * heads * tokens * tokens;
	const int numRowBlocks = (tokens + 15) / 16;
	const int numDBlocks = (headDim + 15) / 16;
	if(numDBlocks > kMaxValueBlocks || numDBlocks <= 0) return;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	if(numWarps <= 0 || numWarps > 8) return;
	extern __shared__ char sharedBytes[];
	const int tileStride = 16 + kSharedMemPad;
	const int paddedTileElements = tileStride * 16;
	auto accumStore = reinterpret_cast<float*>(sharedBytes);
	auto attTiles = reinterpret_cast<__half*>(accumStore + numWarps * paddedTileElements);
	__half* qTiles = attTiles + numWarps * paddedTileElements;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		if(dBlock * 16 >= headDim) break;
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
			const int queryBase = rowBlock * 16;
			if(queryBase >= tokens) continue;
			__half* attTile = attTiles + warpId * paddedTileElements;
			// Load dAtt tile (transposed)
#pragma unroll 4
			for(int idx = laneId; idx < 16 * 16; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalKey = keyStart + r;
				const int globalQuery = queryBase + c;
				float val = 0.0f;
				if(globalKey < tokens && globalQuery < tokens){
					const size_t attIdx = attOffset + globalQuery * tokens + globalKey;
					if(attIdx < totalAttElements){ val = dAtt[attIdx]; }
				}
				attTile[c * tileStride + r] = __float2half(val);
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			load_matrix_sync(attFrag, attTile, tileStride);
			__half* qTile = qTiles + warpId * paddedTileElements;
			// Load Q tile
#pragma unroll 4
			for(int idx = laneId; idx < 16 * 16; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = queryBase + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalQuery < tokens && globalCol < headDim){
					const size_t qIdx = embOffset + globalQuery * headDim + globalCol;
					if(qIdx < totalEmbElements){ val = Q[qIdx]; }
				}
				qTile[r * tileStride + c] = val;
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> qFrag;
			load_matrix_sync(qFrag, qTile, tileStride);
			mma_sync(warpAcc, attFrag, qFrag, warpAcc);
			__syncwarp();
		}
		store_matrix_sync(accumStore + warpId * paddedTileElements, warpAcc, tileStride, wmma::mem_row_major);
		__syncthreads();
		// Reduce and store dK
#pragma unroll 4
		for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			float sum = 0.0f;
#pragma unroll
			for(int w = 0; w < numWarps; ++w){ sum += accumStore[w * paddedTileElements + r * tileStride + c]; }
			const int globalRow = keyStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < tokens && globalCol < headDim){
				const size_t dkIdx = embOffset + globalRow * headDim + globalCol;
				if(dkIdx < totalEmbElements){ dK[dkIdx] = __float2half(sum * scale); }
			}
		}
		__syncthreads();
	}
}
// ============================================================================
// WRAPPER FUNCTIONS
// ============================================================================
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, float* AttentionWeights, int batchSize, int tokens, int headDim, int heads){
	// Validate dimensions
	size_t sharedMemRequired;
	if(!ValidateAttentionDimensions(batchSize, tokens, headDim, heads, sharedMemRequired)){
		printf("WmmaAttention: Invalid dimensions, aborting\n");
		return;
	}
	// Check for null pointers
	if(!Q || !K || !V || !Out){
		printf("WmmaAttention: Null input/output pointer(s)\n");
		return;
	}
	// Configure kernel launch
	const int threadsPerBlock = kDefaultThreads;
	const int numRowBlocks = (tokens + 15) / 16;
	// Validate grid dimensions
	if(numRowBlocks > 65535 || batchSize > 65535 || heads > 65535){
		printf("WmmaAttention: Grid dimensions exceed limits (blocks=%d, batch=%d, heads=%d)\n", numRowBlocks, batchSize, heads);
		return;
	}
	dim3 block(threadsPerBlock);
	dim3 grid(numRowBlocks, batchSize, heads);
	const int tileCols = GetAttentionTileCols(tokens);
	// Set shared memory configuration
	cudaError_t err = cudaFuncSetAttribute(WmmaAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxSharedMemory);
	if(err != cudaSuccess){
		printf("WmmaAttention: Failed to set shared memory size: %s\n", cudaGetErrorString(err));
		return;
	}
	// Set optimal shared memory bank configuration
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	// Launch kernel
	WmmaAttentionKernel<<<grid, block, sharedMemRequired>>>(Q, K, V, Out, AttentionWeights, batchSize, tokens, headDim, heads, tileCols);
	// Check for launch errors
	err = cudaGetLastError();
	if(err != cudaSuccess){ printf("WmmaAttention Forward error: %s\n", cudaGetErrorString(err)); }
}
void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* Att, __half* dQ, __half* dK, __half* dV, float* dAttWorkspace, size_t workspaceElements, int batchSize, int tokens, int headDim, int heads){
	// Validate pointers
	if(!Q || !K || !V || !dOut || !Att || !dQ || !dK || !dV || !dAttWorkspace){
		printf("WmmaAttentionBackward: Null pointer(s) provided\n");
		return;
	}
	// Validate workspace
	const size_t requiredElements = static_cast<size_t>(batchSize) * heads * tokens * tokens;
	if(requiredElements > workspaceElements){
		printf("WmmaAttentionBackward: Workspace too small (%zu required, %zu provided)\n", requiredElements, workspaceElements);
		return;
	}
	// Validate dimensions
	size_t sharedMemRequired;
	if(!ValidateAttentionDimensions(batchSize, tokens, headDim, heads, sharedMemRequired)){
		printf("WmmaAttentionBackward: Invalid dimensions\n");
		return;
	}
	const int numRowBlocks = (tokens + 15) / 16;
	const int numKeyBlocks = (tokens + 15) / 16;
	const int tileCols = GetAttentionTileCols(tokens);
	// Validate grid dimensions
	if(numRowBlocks > 65535 || numKeyBlocks > 65535 || batchSize > 65535 || heads > 65535){
		printf("WmmaAttentionBackward: Grid dimensions exceed limits\n");
		return;
	}
	dim3 block(kDefaultThreads);
	dim3 gridDQ(numRowBlocks, batchSize, heads);
	dim3 gridKV(numKeyBlocks, batchSize, heads);
	const int warpCount = block.x / 32;
	const int tileStride = 16 + kSharedMemPad;
	const size_t paddedTileElements = static_cast<size_t>(tileStride) * 16;
	const size_t smemDQ = sizeof(float) * (warpCount * paddedTileElements + 16) + sizeof(__half) * ((kMaxValueBlocks + 2 * warpCount) * paddedTileElements);
	const size_t smemKV = sizeof(float) * (warpCount * paddedTileElements) + sizeof(__half) * (2 * warpCount * paddedTileElements);
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
// ============================================================================
// PACKING/UNPACKING UTILITIES
// ============================================================================
__global__ void PackColumnsToHeadsKernel(const __half* __restrict__ input, __half* __restrict__ output, int B, int T, int H, int D){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = B * T * H * D;
	if(idx >= total) return;
	// Compute indices
	const int d = idx % D;
	int tmp = idx / D;
	const int t = tmp % T;
	tmp /= T;
	const int h = tmp % H;
	const int b = tmp / H;
	const int embedDim = H * D;
	const int col = b * T + t;
	const int row = h * D + d;
	// Bounds checking
	const size_t inIdx = static_cast<size_t>(row) + col * embedDim;
	const size_t maxIdx = static_cast<size_t>(B) * T * embedDim;
	if(inIdx < maxIdx){ output[idx] = input[inIdx]; } else{ output[idx] = __float2half(0.0f); }
}
void PackColumnsToHeads(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads){
	if(!input || !output){
		printf("PackColumnsToHeads: Null pointer(s)\n");
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
	if(batch <= 0 || batch > 1024 || tokens <= 0 || tokens > kMaxTokens){
		printf("PackColumnsToHeads: Invalid dimensions B=%d, T=%d\n", batch, tokens);
		return;
	}
	const int headDim = embedDim / numHeads;
	const int total = batch * tokens * embedDim;
	constexpr int blockSize = 256;
	const int numBlocks = DivCeil(total, blockSize);
	if(numBlocks > 0 && numBlocks <= 65535){
		PackColumnsToHeadsKernel<<<numBlocks, blockSize>>>(input, output, batch, tokens, numHeads, headDim);
		cudaError_t err = cudaGetLastError();
		if(err != cudaSuccess){ printf("PackColumnsToHeads error: %s\n", cudaGetErrorString(err)); }
	} else{ printf("PackColumnsToHeads: Invalid grid size %d\n", numBlocks); }
}
__global__ void PackHeadsToColumnsKernel(const __half* __restrict__ input, __half* __restrict__ output, int B, int T, int H, int D){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = B * T * H * D;
	if(idx >= total) return;
	// Compute indices
	const int d = idx % D;
	int tmp = idx / D;
	const int t = tmp % T;
	tmp /= T;
	const int h = tmp % H;
	const int b = tmp / H;
	const int embedDim = H * D;
	const int col = b * T + t;
	const int row = h * D + d;
	// Bounds checking
	const size_t outIdx = static_cast<size_t>(row) + col * embedDim;
	const size_t maxIdx = static_cast<size_t>(B) * T * embedDim;
	if(outIdx < maxIdx){ output[outIdx] = input[idx]; }
}
void PackHeadsToColumns(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads){
	if(!input || !output){
		printf("PackHeadsToColumns: Null pointer(s)\n");
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
	if(batch <= 0 || batch > 1024 || tokens <= 0 || tokens > kMaxTokens){
		printf("PackHeadsToColumns: Invalid dimensions B=%d, T=%d\n", batch, tokens);
		return;
	}
	const int headDim = embedDim / numHeads;
	const int total = batch * tokens * embedDim;
	constexpr int blockSize = 256;
	const int numBlocks = DivCeil(total, blockSize);
	if(numBlocks > 0 && numBlocks <= 65535){
		PackHeadsToColumnsKernel<<<numBlocks, blockSize>>>(input, output, batch, tokens, numHeads, headDim);
		cudaError_t err = cudaGetLastError();
		if(err != cudaSuccess){ printf("PackHeadsToColumns error: %s\n", cudaGetErrorString(err)); }
	} else{ printf("PackHeadsToColumns: Invalid grid size %d\n", numBlocks); }
}