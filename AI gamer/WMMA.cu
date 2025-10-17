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
	// Increased padding to better avoid bank conflicts
	constexpr int kSharedMemPad = 8;
	// FTZ threshold for numerical stability
	constexpr float SOFTMAX_FTZ_THRESHOLD = -12.0f;
	int GetAttentionTileCols(const int T){
		int limited = T;
		if(limited < 16) limited = 16;
		if(limited > kMaxTileCols) limited = kMaxTileCols;
		const int remainder = limited % 16;
		if(remainder != 0) limited += 16 - remainder;
		return limited;
	}
	// Helper for warp-level reduction
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
}
__global__ void WmmaAttentionKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ Out, float* __restrict__ AttentionWeights, int B, int T, int D, int H, int tileCols){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	if(rowBlock * 16 >= T) return;
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	const int batchHeadOffset = (batch * H + head) * T * D;
	extern __shared__ char sharedMemBytes[];
	// Optimized strides with better padding
	const int qStride = D + kSharedMemPad;
	const int tileStride = 16 + kSharedMemPad;
	auto qShared = reinterpret_cast<__half*>(sharedMemBytes);
	__half* warpTiles = qShared + 16 * qStride;
	__half* attTile = warpTiles + numWarps * tileStride * 16;
	auto scoresTile = reinterpret_cast<float*>(attTile + tileStride * 16);
	float* rowMax = scoresTile + 16 * tileCols;
	float* rowSum = rowMax + 16;
	const float scale = rsqrtf(static_cast<float>(D));
	const __half scaleHalf = __float2half(scale);
	const __half2 scaleHalf2 = __float2half2_rn(scale);
	const __half zeroHalf = __float2half(0.0f);
	const int qBlocks = (D + 15) / 16;
	if(qBlocks > kMaxValueBlocks){
		if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){ printf("WmmaAttention: q blocks %d exceed limit %d\n", qBlocks, kMaxValueBlocks); }
		return;
	}
	// Q fragments that will be reused throughout
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frags[kMaxValueBlocks];
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
	// Initialize row stats
	if(threadIdx.x < 16){
		rowMax[threadIdx.x] = -FLT_MAX;
		rowSum[threadIdx.x] = 0.0f;
	}
	__syncthreads();
	// Load Q rows cooperatively with alignment-safe vectorization
	for(int row = 0; row < 16; ++row){
		const int globalRow = rowBlock * 16 + row;
		__half* sharedRow = qShared + row * qStride;
		for(int col = threadIdx.x; col < qStride; col += blockDim.x){ sharedRow[col] = __float2half(0.0f); }
		if(globalRow >= T){ continue; }
		const __half* qRow = Q + batchHeadOffset + globalRow * D;
		int sharedOffset = 0;
		int remaining = D;
		if(remaining > 0 && (reinterpret_cast<uintptr_t>(qRow) & 0x3)){ // handle leading misalignment
			if(threadIdx.x == 0){ sharedRow[0] = __hmul(qRow[0], scaleHalf); }
			qRow += 1;
			sharedOffset = 1;
			remaining -= 1;
		}
		const int pairCount = remaining / 2;
		if(pairCount > 0){
			auto qRow2 = reinterpret_cast<const __half2*>(qRow);
			auto sharedRow2 = reinterpret_cast<__half2*>(sharedRow + sharedOffset);
			for(int pairIdx = threadIdx.x; pairIdx < pairCount; pairIdx += blockDim.x){
				__half2 val = __ldg(qRow2 + pairIdx);
				sharedRow2[pairIdx] = __hmul2(val, scaleHalf2);
			}
		}
		if((remaining & 1) != 0){
			if(threadIdx.x == 0){ sharedRow[sharedOffset + pairCount * 2] = __hmul(qRow[pairCount * 2], scaleHalf); }
		}
	}
	__syncthreads();
	// Load Q into fragments ONCE and reuse throughout
#pragma unroll
	for(int dBlock = 0; dBlock < qBlocks; ++dBlock){ load_matrix_sync(q_frags[dBlock], qShared + dBlock * 16, qStride); }
	const int attentionOffset = (batch * H + head) * T * T;
	// Process K tiles with improved memory access pattern
	for(int tileStart = 0; tileStart < T; tileStart += tileCols){
		const int remaining = T - tileStart;
		const int tileWidth = (remaining > tileCols) ? tileCols : remaining;
		if(tileWidth <= 0) break;
		// Compute QK^T scores
		for(int colBlock = 0; colBlock < tileWidth; colBlock += 16 * numWarps){
			const int remainingCols = tileWidth - colBlock;
			int activeWarps = 0;
			if(remainingCols > 0){
				activeWarps = (remainingCols + 15) / 16;
				if(activeWarps > numWarps) activeWarps = numWarps;
			}
			wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpScores;
			if(warpId < activeWarps){ fill_fragment(warpScores, 0.0f); }
			// Process K blocks
#pragma unroll 2
			for(int kBlock = 0; kBlock < qBlocks; kBlock++){
				// Collaborative K tile loading with vectorized access
				const int elementsPerTile = 16 * 16;
				const int pairsPerTile = elementsPerTile / 2;
				const int totalPairs = pairsPerTile * activeWarps;
#pragma unroll 4
				for(int idx = threadIdx.x; idx < totalPairs; idx += blockDim.x){
					const int warpLocal = idx / pairsPerTile;
					const int pairIndex = idx % pairsPerTile;
					const int col = pairIndex / 8;
					const int rowPair = (pairIndex % 8) * 2;
					const int localCol = warpLocal * 16 + col;
					const int globalCol = tileStart + colBlock + localCol;
					auto tileBase = warpTiles + warpLocal * tileStride * 16 + col * tileStride;
					if(localCol >= remainingCols || globalCol >= T){
						tileBase[rowPair] = zeroHalf;
						tileBase[rowPair + 1] = zeroHalf;
						continue;
					}
					const int globalRow = kBlock * 16 + rowPair;
					if(globalRow >= D){
						tileBase[rowPair] = zeroHalf;
						tileBase[rowPair + 1] = zeroHalf;
						continue;
					}
					const __half* kPtr = K + batchHeadOffset + globalCol * D + kBlock * 16;
					const __half* rowPtr = kPtr + rowPair;
					const bool alignedLoad = ((reinterpret_cast<uintptr_t>(rowPtr) & 0x3u) == 0u) && ((reinterpret_cast<uintptr_t>(tileBase + rowPair) & 0x3u) == 0u);
					if(globalRow + 1 < D && alignedLoad){
						reinterpret_cast<__half2*>(tileBase + rowPair)[0] = reinterpret_cast<const __half2*>(rowPtr)[0];
					} else{
						tileBase[rowPair] = rowPtr[0];
						if(globalRow + 1 < D){ tileBase[rowPair + 1] = rowPtr[1]; } else{ tileBase[rowPair + 1] = zeroHalf; }
					}
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
		// Optimized softmax computation
#pragma unroll 4
		for(int row = warpId; row < 16; row += numWarps){
			const int globalRow = rowBlock * 16 + row;
			if(globalRow >= T) continue;
			const float prevMax = rowMax[row];
			const float prevSum = rowSum[row];
			// Find max with warp reduction
			float localMax = -FLT_MAX;
#pragma unroll 4
			for(int col = laneId; col < tileWidth; col += 32){
				const float val = scoresTile[row * tileCols + col];
				localMax = fmaxf(localMax, val);
			}
			localMax = WarpReduceMax(localMax);
			const float newMax = fmaxf(prevMax, localMax);
			// Compute exp sum with numerical stability
			const float scalePrev = (prevSum > 0.0f) ? expf(prevMax - newMax) : 0.0f;
			float localSum = 0.0f;
#pragma unroll 4
			for(int col = laneId; col < tileWidth; col += 32){
				const float val = scoresTile[row * tileCols + col];
				const float exp_val = expf(val - newMax);
				// Apply FTZ for very small values
				localSum += (val - newMax > SOFTMAX_FTZ_THRESHOLD) ? exp_val : 0.0f;
			}
			localSum = WarpReduceSum(localSum);
			if(laneId == 0){
				rowMax[row] = newMax;
				rowSum[row] = prevSum * scalePrev + localSum;
			}
		}
		__syncthreads();
	}
	// Final softmax normalization
	const int valueBlocks = (D + 15) / 16;
	if(valueBlocks > kMaxValueBlocks || valueBlocks == 0) return;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frags[kMaxValueBlocks];
#pragma unroll
	for(int vb = warpId; vb < valueBlocks; vb += numWarps){ fill_fragment(out_frags[vb], 0.0f); }
	// Process V tiles
	for(int tileStart = 0; tileStart < T; tileStart += tileCols){
		const int remaining = T - tileStart;
		const int tileWidth = (remaining > tileCols) ? tileCols : remaining;
		if(tileWidth <= 0) break;
		// Apply softmax normalization
#pragma unroll 4
		for(int row = warpId; row < 16; row += numWarps){
			const int globalRow = rowBlock * 16 + row;
			if(globalRow >= T) continue;
			const float maxVal = rowMax[row];
			const float denom = rowSum[row];
			const float invDenom = (denom > 0.0f) ? (1.0f / denom) : 0.0f;
#pragma unroll 4
			for(int col = laneId; col < tileWidth; col += 32){
				const int globalCol = tileStart + col;
				float logits = scoresTile[row * tileCols + col];
				float normalized = 0.0f;
				if(globalCol < T){
					const float exp_val = expf(logits - maxVal);
					// Apply FTZ
					normalized = (logits - maxVal > SOFTMAX_FTZ_THRESHOLD) ? exp_val * invDenom : 0.0f;
				}
				scoresTile[row * tileCols + col] = normalized;
				if(AttentionWeights){ AttentionWeights[attentionOffset + globalRow * T + globalCol] = normalized; }
			}
		}
		__syncthreads();
		// Compute attention * V
		for(int colBlock = 0; colBlock < tileWidth; colBlock += 16){
			// Load attention tile
#pragma unroll 4
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int row = idx / 16;
				const int col = idx % 16;
				const int localCol = colBlock + col;
				const int globalCol = tileStart + localCol;
				float val = 0.0f;
				if(rowBlock * 16 + row < T && localCol < tileWidth && globalCol < T){ val = scoresTile[row * tileCols + localCol]; }
				attTile[row * tileStride + col] = __float2half(val);
			}
			__syncthreads();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> att_frag;
			load_matrix_sync(att_frag, attTile, tileStride);
			// Process V blocks
			const int valueWarps = (numWarps < valueBlocks) ? numWarps : valueBlocks;
			for(int vbBase = 0; vbBase < valueBlocks; vbBase += valueWarps){
				const int activeValueWarps = ((valueBlocks - vbBase) > valueWarps) ? valueWarps : (valueBlocks - vbBase);
				// Load V tiles with vectorization
				const int elementsPerTile = 16 * 16;
				const int pairsPerTile = elementsPerTile / 2;
				const int totalPairs = pairsPerTile * activeValueWarps;
#pragma unroll 4
				for(int idx = threadIdx.x; idx < totalPairs; idx += blockDim.x){
					const int warpLocal = idx / pairsPerTile;
					const int pairIndex = idx % pairsPerTile;
					const int row = pairIndex / 8;
					const int colPair = (pairIndex % 8) * 2;
					const int keyIdx = tileStart + colBlock + row;
					const int vb = vbBase + warpLocal;
					const int valueBase = vb * 16 + colPair;
					auto tileBase = warpTiles + warpLocal * tileStride * 16 + row * tileStride;
					if(row >= tileWidth || keyIdx >= T || vb >= valueBlocks){
						tileBase[colPair] = zeroHalf;
						tileBase[colPair + 1] = zeroHalf;
						continue;
					}
					if(valueBase >= D){
						tileBase[colPair] = zeroHalf;
						tileBase[colPair + 1] = zeroHalf;
						continue;
					}
					const __half* vPtr = V + batchHeadOffset + keyIdx * D + vb * 16;
					const __half* rowPtr = vPtr + colPair;
					const bool alignedLoad = ((reinterpret_cast<uintptr_t>(rowPtr) & 0x3u) == 0u) && ((reinterpret_cast<uintptr_t>(tileBase + colPair) & 0x3u) == 0u);
					if(valueBase + 1 < D && alignedLoad){
						reinterpret_cast<__half2*>(tileBase + colPair)[0] = reinterpret_cast<const __half2*>(rowPtr)[0];
					} else{
						tileBase[colPair] = rowPtr[0];
						if(valueBase + 1 < D){ tileBase[colPair + 1] = rowPtr[1]; } else{ tileBase[colPair + 1] = zeroHalf; }
					}
				}
				__syncthreads();
				if(warpId < activeValueWarps){
					const int vb = vbBase + warpId;
					load_matrix_sync(v_frag, warpTiles + warpId * tileStride * 16, tileStride);
					mma_sync(out_frags[vb], att_frag, v_frag, out_frags[vb]);
				}
				__syncthreads();
			}
		}
	}
	// Store output with vectorization where possible
	for(int vb = warpId; vb < valueBlocks; vb += numWarps){
#pragma unroll
		for(int i = 0; i < out_frags[vb].num_elements; i++){
			const int row = i / 16;
			const int col = i % 16;
			const int globalRow = rowBlock * 16 + row;
			const int globalCol = vb * 16 + col;
			if(globalRow < T && globalCol < D){ Out[batchHeadOffset + globalRow * D + globalCol] = __float2half(out_frags[vb].x[i]); }
		}
	}
}
// Updated wrapper function to use optimized kernel
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, float* AttentionWeights, int B, int T, int D, int H){
	// Use 256 threads for better occupancy
	dim3 block(256);
	dim3 grid((T + 15) / 16, B, H);
	const int tileCols = GetAttentionTileCols(T);
	const int warpCount = block.x / 32;
	const int qStride = D + kSharedMemPad;
	const int tileStride = 16 + kSharedMemPad;
	size_t sharedSize = sizeof(__half) * (16 * qStride + warpCount * tileStride * 16 + tileStride * 16) + sizeof(float) * (16 * tileCols + 16 + 16);
	cudaFuncSetAttribute(WmmaAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	WmmaAttentionKernel<<<grid, block, sharedSize>>>(Q, K, V, Out, AttentionWeights, B, T, D, H, tileCols);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess){ printf("WmmaAttention Forward error: %s\n", cudaGetErrorString(e)); }
}
__global__ void ComputeDAttDQKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ attention, float* __restrict__ dAtt, __half* __restrict__ dQ, int B, int T, int D, int H, int tileCols){
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
	if(numDBlocks > kMaxValueBlocks){
		if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){ printf("ComputeDAttDQ: d blocks %d exceed limit %d\n", numDBlocks, kMaxValueBlocks); }
		return;
	}
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	extern __shared__ char sharedBytes[];
	const int elementsPerTile = 16 * 16;
	const int tileStride = 16 + kSharedMemPad;
	const int paddedTileElements = tileStride * 16;
	auto scoreTiles = reinterpret_cast<float*>(sharedBytes);
	float* rowSums = scoreTiles + numWarps * paddedTileElements;
	auto outTiles = reinterpret_cast<__half*>(rowSums + 16);
	__half* valueTiles = outTiles + kMaxValueBlocks * paddedTileElements;
	__half* attTiles = valueTiles + numWarps * paddedTileElements;
	if(threadIdx.x < 16){ rowSums[threadIdx.x] = 0.0f; }
	__syncthreads();
	const uintptr_t dOutAlignedPtr = reinterpret_cast<uintptr_t>(dOut + embOffset);
	const uintptr_t outTilesAlignedPtr = reinterpret_cast<uintptr_t>(outTiles);
	const bool canVectorizeOutTiles = ((dOutAlignedPtr | outTilesAlignedPtr) & 0x3) == 0 && ((D & 1) == 0);
	const uintptr_t valueTilesAlignedPtr = reinterpret_cast<uintptr_t>(valueTiles);
	const uintptr_t vAlignedPtr = reinterpret_cast<uintptr_t>(V + embOffset);
	const bool canVectorizeValueTiles = ((valueTilesAlignedPtr | vAlignedPtr) & 0x3u) == 0u && ((D & 1) == 0) && (D >= 2);
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		const int colBase = dBlock * 16;
		int validCols = D - colBase;
		if(validCols > 16){ validCols = 16; }
		if(validCols < 0){ validCols = 0; }
		__half* outTile = outTiles + dBlock * paddedTileElements;
		if(canVectorizeOutTiles){
			const int vectorCols = validCols & ~1;
			const int vectorPairs = vectorCols / 2;
			const int tileStridePairs = tileStride / 2;
			const int paddedTilePairs = paddedTileElements / 2;
			auto outTilePairs = reinterpret_cast<__half2*>(outTile);
			const __half2 zeroPair = __float2half2_rn(0.0f);
#pragma unroll
			for(int idx = threadIdx.x; idx < paddedTilePairs; idx += blockDim.x){
				const int row = idx / tileStridePairs;
				const int pairIdx = idx % tileStridePairs;
				if(row >= 16) continue;
				const int globalRow = rowStart + row;
				__half2 val = zeroPair;
				if(globalRow < T && pairIdx < vectorPairs){
					const __half2* rowSrcPairs = reinterpret_cast<const __half2*>(dOut + embOffset + globalRow * D + colBase);
					val = rowSrcPairs[pairIdx];
				}
				outTilePairs[idx] = val;
			}
			__syncthreads();
			if((validCols & 1) != 0){
				const int tailCol = vectorCols;
				if(threadIdx.x < 16){
					const int globalRow = rowStart + threadIdx.x;
					const int globalCol = colBase + tailCol;
					__half tailVal = __float2half(0.0f);
					if(globalRow < T && globalCol < D){ tailVal = dOut[embOffset + globalRow * D + globalCol]; }
					outTile[threadIdx.x * tileStride + tailCol] = tailVal;
				}
			}
		} else{
#pragma unroll
			for(int idx = threadIdx.x; idx < paddedTileElements; idx += blockDim.x){
				const int row = idx / tileStride;
				const int col = idx % tileStride;
				if(row >= 16) continue;
				const int globalRow = rowStart + row;
				const int globalCol = colBase + col;
				__half val = __float2half(0.0f);
				if(globalRow < T && col < 16 && globalCol < D){ val = dOut[embOffset + globalRow * D + globalCol]; }
				outTile[row * tileStride + col] = val;
			}
		}
		__syncthreads();
	}
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
				const __half* outTile = outTiles + dBlock * paddedTileElements;
				wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> outFrag;
				load_matrix_sync(outFrag, outTile, tileStride);
				if(canVectorizeValueTiles){
					constexpr int kRowPairsPerTile = 8;
					constexpr int kColPairsPerTile = 8;
					constexpr int kVecsPerTile = kRowPairsPerTile * kColPairsPerTile;
					const int totalVecs = activeWarps * kVecsPerTile;
					for(int vecIdx = threadIdx.x; vecIdx < totalVecs; vecIdx += blockDim.x){
						const int warpLocal = vecIdx / kVecsPerTile;
						const int pairIndex = vecIdx % kVecsPerTile;
						const int rowPair = (pairIndex / kColPairsPerTile) * 2;
						const int colPair = (pairIndex % kColPairsPerTile) * 2;
						const int warpColBase = warpLocal * 16;
						int validRows = remainingCols - warpColBase;
						if(validRows < 0) validRows = 0;
						if(validRows > 16) validRows = 16;
						const int globalRow0 = tileStart + colBlock + warpColBase + rowPair;
						const int globalRow1 = globalRow0 + 1;
						const int globalCol0 = dBlock * 16 + colPair;
						const int globalCol1 = globalCol0 + 1;
						const bool row0Valid = globalRow0 < T;
						const bool row1Valid = globalRow1 < T;
						const bool row0InTile = rowPair < validRows;
						const bool row1InTile = (rowPair + 1) < validRows;
						const bool col0Valid = globalCol0 < D;
						const bool col1Valid = globalCol1 < D;
						__half row0Col0 = __half(0.0f);
						__half row1Col0 = __half(0.0f);
						__half row0Col1 = __half(0.0f);
						__half row1Col1 = __half(0.0f);
						if(col0Valid && row0Valid && row0InTile){
							const __half* rowPtr = V + embOffset + globalRow0 * D + globalCol0;
							if(col1Valid){
								const __half2 vec = reinterpret_cast<const __half2*>(rowPtr)[0];
								row0Col0 = __low2half(vec);
								row0Col1 = __high2half(vec);
							} else{ row0Col0 = rowPtr[0]; }
						}
						if(col0Valid && row1Valid && row1InTile){
							const __half* rowPtr = V + embOffset + globalRow1 * D + globalCol0;
							if(col1Valid){
								const __half2 vec = reinterpret_cast<const __half2*>(rowPtr)[0];
								row1Col0 = __low2half(vec);
								row1Col1 = __high2half(vec);
							} else{ row1Col0 = rowPtr[0]; }
						}
						auto warpBasePtr = valueTiles + warpLocal * paddedTileElements;
						auto col0Ptr = reinterpret_cast<__half2*>(warpBasePtr + colPair * tileStride + rowPair);
						col0Ptr[0] = __halves2half2(row0Col0, row1Col0);
						auto col1Ptr = reinterpret_cast<__half2*>(warpBasePtr + (colPair + 1) * tileStride + rowPair);
						col1Ptr[0] = __halves2half2(row0Col1, row1Col1);
					}
				} else{
#pragma unroll
					for(int idx = threadIdx.x; idx < activeWarps * elementsPerTile; idx += blockDim.x){
						const int warpLocal = idx / elementsPerTile;
						const int tileIndex = idx % elementsPerTile;
						const int r = tileIndex / 16;
						const int c = tileIndex % 16;
						const int warpColBase = warpLocal * 16;
						const int globalRow = tileStart + colBlock + warpColBase + r;
						const int globalCol = dBlock * 16 + c;
						__half val = __float2half(0.0f);
						if(globalRow < T && globalCol < D && warpColBase + r < remainingCols){ val = V[embOffset + globalRow * D + globalCol]; }
						valueTiles[warpLocal * paddedTileElements + c * tileStride + r] = val;
					}
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
				if(globalRow < T && c < validCols && globalCol < T){ dAtt[attOffset + globalRow * T + globalCol] = scoreTiles[warpLocal * paddedTileElements + r * tileStride + c]; }
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
					const float* tile = scoreTiles + warpLocal * paddedTileElements + row * tileStride;
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
			__half* attTile = attTiles + warpId * paddedTileElements;
			__half* kTile = valueTiles + warpId * paddedTileElements;
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = rowStart + r;
				const int globalCol = keyBase + c;
				float val = 0.0f;
				if(globalRow < T && globalCol < T){ val = dAtt[attOffset + globalRow * T + globalCol]; }
				attTile[r * tileStride + c] = __float2half(val);
			}
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = keyBase + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalRow < T && globalCol < D){ val = K[embOffset + globalRow * D + globalCol]; }
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
		for(int idx = threadIdx.x; idx < elementsPerTile; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			float sum = 0.0f;
			for(int w = 0; w < numWarps; ++w){ sum += scoreTiles[w * paddedTileElements + r * tileStride + c]; }
			const int globalRow = rowStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < T && globalCol < D){ dQ[embOffset + globalRow * D + globalCol] = __float2half(sum * scale); }
		}
		__syncthreads();
	}
}
__global__ void ComputeDVKernel(const float* __restrict__ attention, const __half* __restrict__ dOut, __half* __restrict__ dV, int B, int T, int D, int H){
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
	if(numDBlocks > kMaxValueBlocks){
		if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){ printf("ComputeDV: d blocks %d exceed limit %d\n", numDBlocks, kMaxValueBlocks); }
		return;
	}
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	extern __shared__ char sharedBytes[];
	const int elementsPerTile = 16 * 16;
	const int tileStride = 16 + kSharedMemPad;
	const int paddedTileElements = tileStride * 16;
	auto accumStore = reinterpret_cast<float*>(sharedBytes);
	auto attTiles = reinterpret_cast<__half*>(accumStore + numWarps * paddedTileElements);
	__half* outTiles = attTiles + numWarps * paddedTileElements;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
			const int queryBase = rowBlock * 16;
			__half* attTile = attTiles + warpId * paddedTileElements;
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = queryBase + r;
				const int globalKey = keyStart + c;
				float val = 0.0f;
				if(globalQuery < T && globalKey < T){ val = attention[attOffset + globalQuery * T + globalKey]; }
				attTile[c * tileStride + r] = __float2half(val);
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			load_matrix_sync(attFrag, attTile, tileStride);
			__half* outTile = outTiles + warpId * paddedTileElements;
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = queryBase + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalQuery < T && globalCol < D){ val = dOut[embOffset + globalQuery * D + globalCol]; }
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
		for(int idx = threadIdx.x; idx < elementsPerTile; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			float sum = 0.0f;
			for(int w = 0; w < numWarps; ++w){ sum += accumStore[w * paddedTileElements + r * tileStride + c]; }
			const int globalRow = keyStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < T && globalCol < D){ dV[embOffset + globalRow * D + globalCol] = __float2half(sum); }
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
	if(numDBlocks > kMaxValueBlocks){
		if(threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){ printf("ComputeDK: d blocks %d exceed limit %d\n", numDBlocks, kMaxValueBlocks); }
		return;
	}
	const float scale = rsqrtf(static_cast<float>(D));
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	extern __shared__ char sharedBytes[];
	const int elementsPerTile = 16 * 16;
	const int tileStride = 16 + kSharedMemPad;
	const int paddedTileElements = tileStride * 16;
	auto accumStore = reinterpret_cast<float*>(sharedBytes);
	auto attTiles = reinterpret_cast<__half*>(accumStore + numWarps * paddedTileElements);
	__half* qTiles = attTiles + numWarps * paddedTileElements;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> warpAcc;
		fill_fragment(warpAcc, 0.0f);
		for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
			const int queryBase = rowBlock * 16;
			__half* attTile = attTiles + warpId * paddedTileElements;
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalKey = keyStart + r;
				const int globalQuery = queryBase + c;
				float val = 0.0f;
				if(globalKey < T && globalQuery < T){ val = dAtt[attOffset + globalQuery * T + globalKey]; }
				attTile[c * tileStride + r] = __float2half(val);
			}
			__syncwarp();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			load_matrix_sync(attFrag, attTile, tileStride);
			__half* qTile = qTiles + warpId * paddedTileElements;
			for(int idx = laneId; idx < elementsPerTile; idx += 32){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = queryBase + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalQuery < T && globalCol < D){ val = Q[embOffset + globalQuery * D + globalCol]; }
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
		for(int idx = threadIdx.x; idx < elementsPerTile; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			float sum = 0.0f;
			for(int w = 0; w < numWarps; ++w){ sum += accumStore[w * paddedTileElements + r * tileStride + c]; }
			const int globalRow = keyStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < T && globalCol < D){ dK[embOffset + globalRow * D + globalCol] = __float2half(sum * scale); }
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
	const int tileStride = 16 + kSharedMemPad;
	const size_t paddedTileElements = static_cast<size_t>(tileStride) * 16;
	const size_t smemDQ = sizeof(float) * (warpCount * paddedTileElements + 16) + sizeof(__half) * ((kMaxValueBlocks + 2 * warpCount) * paddedTileElements);
	const size_t smemDV = sizeof(float) * (warpCount * paddedTileElements) + sizeof(__half) * (2 * warpCount * paddedTileElements);
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
__global__ void PackColumnsToHeadsKernel(const __half* __restrict__ input, __half* __restrict__ output, int B, int T, int H, int D){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = B * T * H * D;
	if(idx >= total) return;
	const int d = idx % D;
	int tmp = idx / D;
	const int t = tmp % T;
	tmp /= T;
	const int h = tmp % H;
	const int b = tmp / H;
	const int embedDim = H * D;
	const int col = t * B + b;
	const int row = h * D + d;
	output[idx] = input[row + col * embedDim];
}
void PackColumnsToHeads(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads){
	if(numHeads <= 0) return;
	if(embedDim % numHeads != 0){
		printf("PackColumnsToHeads embedDim %d not divisible by numHeads %d\n", embedDim, numHeads);
		return;
	}
	const int headDim = embedDim / numHeads;
	const int total = batch * tokens * embedDim;
	constexpr int bs = 256;
	const int blocks = DivCeil(total, bs);
	PackColumnsToHeadsKernel<<<blocks, bs>>>(input, output, batch, tokens, numHeads, headDim);
	const auto err = cudaGetLastError();
	if(err != cudaSuccess){ printf("PackColumnsToHeads error: %s\n", cudaGetErrorString(err)); }
}
__global__ void PackHeadsToColumnsKernel(const __half* __restrict__ input, __half* __restrict__ output, int B, int T, int H, int D){
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int total = B * T * H * D;
	if(idx >= total) return;
	const int d = idx % D;
	int tmp = idx / D;
	const int t = tmp % T;
	tmp /= T;
	const int h = tmp % H;
	const int b = tmp / H;
	const int embedDim = H * D;
	const int col = t * B + b;
	const int row = h * D + d;
	output[row + col * embedDim] = input[idx];
}
void PackHeadsToColumns(const __half* input, __half* output, int batch, int tokens, int embedDim, int numHeads){
	if(numHeads <= 0) return;
	if(embedDim % numHeads != 0){
		printf("PackHeadsToColumns embedDim %d not divisible by numHeads %d\n", embedDim, numHeads);
		return;
	}
	const int headDim = embedDim / numHeads;
	const int total = batch * tokens * embedDim;
	constexpr int bs = 256;
	const int blocks = DivCeil(total, bs);
	PackHeadsToColumnsKernel<<<blocks, bs>>>(input, output, batch, tokens, numHeads, headDim);
	const auto err = cudaGetLastError();
	if(err != cudaSuccess){ printf("PackHeadsToColumns error: %s\n", cudaGetErrorString(err)); }
}