// WMMA Attention Optimized for Titan V - Fixed Version
// Properly handles fragment type conversions
#define __CUDACC__
#include "CuCommon.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda;
namespace{
	// Volta-specific constants
	constexpr int kVoltaSharedMemory = 98304; // 96KB
	constexpr int kMaxValueBlocks = 32;
	constexpr int kSharedMemPad = 4;
	constexpr int kOptimalThreadsVolta = 128;
	constexpr int kSharedMemAlignment = 16;
	template <typename T> __host__ __device__ inline T AlignUp(T value, T alignment){ return ((value + alignment - 1) / alignment) * alignment; }
	__device__ inline unsigned char* AlignSharedPtr(unsigned char* ptr, size_t alignment){
		auto addr = reinterpret_cast<uintptr_t>(ptr);
		addr = AlignUp(addr, alignment);
		return reinterpret_cast<unsigned char*>(addr);
	}
}
// ============================================================================
// OPTIMIZED BACKWARD KERNEL - COMPUTE dAtt AND dQ
// ============================================================================
__global__ void __launch_bounds__(256, 2) ComputeDAttDQOptimized(
	const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V,
	const __half* __restrict__ dOut, const __half* __restrict__ Att,
	__half* __restrict__ dQ, float* __restrict__ dAttWorkspace,
	int batchSize, int tokens, int headDim, int heads){

	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || rowBlock * 16 >= tokens) return;

	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	const int rowStart = rowBlock * 16;

	const size_t batchHeadOffset = (static_cast<size_t>(batch) * heads + head) * tokens * headDim;
	const size_t attentionOffset = (static_cast<size_t>(batch) * heads + head) * tokens * tokens;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));

	const int vBlocks = (headDim + 15) / 16;
	const int numKeyBlocks = (tokens + 15) / 16;
	if(vBlocks > kMaxValueBlocks) return;

	extern __shared__ __align__(16) unsigned char sharedStorage[];
	unsigned char* sharedMemBytes = sharedStorage;

	const int tileStride = AlignUp(16 + kSharedMemPad, kSharedMemAlignment);
	const int qStride = AlignUp(headDim + kSharedMemPad, kSharedMemAlignment);

	// Shared memory layout
	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto dOutShared = reinterpret_cast<__half*>(sharedMemBytes);
	sharedMemBytes += sizeof(__half) * 16 * qStride;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto vTiles = reinterpret_cast<__half*>(sharedMemBytes);
	sharedMemBytes += sizeof(__half) * numWarps * 16 * qStride;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto kTiles = reinterpret_cast<__half*>(sharedMemBytes);
	sharedMemBytes += sizeof(__half) * numWarps * 16 * qStride;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto dAttTiles = reinterpret_cast<float*>(sharedMemBytes);
	sharedMemBytes += sizeof(float) * numWarps * tileStride * 16;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto attTiles = reinterpret_cast<__half*>(sharedMemBytes);
	sharedMemBytes += sizeof(__half) * numWarps * tileStride * 16;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	float* rowSums = reinterpret_cast<float*>(sharedMemBytes);
	sharedMemBytes += sizeof(float) * 16;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	float* dQWorkspace = reinterpret_cast<float*>(sharedMemBytes);

	// Initialize row sums
	if(threadIdx.x < 16){ rowSums[threadIdx.x] = 0.0f; }

	// Load dOut to shared memory
	for(int row = 0; row < 16; ++row){
		const int globalRow = rowStart + row;
		__half* sharedRow = dOutShared + row * qStride;
		if(globalRow < tokens){
			const size_t dOutRowOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
			for(int col = threadIdx.x; col < headDim; col += blockDim.x){
				sharedRow[col] = dOut[dOutRowOffset + col];
			}
			for(int col = threadIdx.x + headDim; col < qStride; col += blockDim.x){
				sharedRow[col] = __float2half(0.0f);
			}
		} else{
			for(int col = threadIdx.x; col < qStride; col += blockDim.x){
				sharedRow[col] = __float2half(0.0f);
			}
		}
	}
	__syncthreads();

	// Load dOut fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> dOut_frags[kMaxValueBlocks];
#pragma unroll
	for(int dBlock = 0; dBlock < vBlocks; ++dBlock){
		if(dBlock * 16 < headDim){
			load_matrix_sync(dOut_frags[dBlock], dOutShared + dBlock * 16, qStride);
		}
	}

	// ====== PASS 1: Compute dAtt = dOut @ V^T and accumulate row sums ======
	for(int keyBlock = warpId; keyBlock < numKeyBlocks; keyBlock += numWarps){
		const int keyBase = keyBlock * 16;
		if(keyBase >= tokens) continue;

		// Load V tile for this warp
		__half* myVTile = vTiles + warpId * 16 * qStride;
		for(int row = 0; row < 16; ++row){
			const int globalKey = keyBase + row;
			__half* sharedRow = myVTile + row * qStride;
			if(globalKey < tokens){
				const size_t vOffset = batchHeadOffset + globalKey * headDim;
#pragma unroll 4
				for(int col = laneId; col < headDim; col += 32){
					sharedRow[col] = V[vOffset + col];
				}
			}
			for(int col = laneId + headDim; col < qStride; col += 32){
				sharedRow[col] = __float2half(0.0f);
			}
		}
		__syncwarp();

		// Compute dAtt tile using WMMA
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> dAtt_acc;
		fill_fragment(dAtt_acc, 0.0f);

#pragma unroll
		for(int vBlock = 0; vBlock < vBlocks; ++vBlock){
			if(vBlock * 16 < headDim){
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> v_frag;
				load_matrix_sync(v_frag, myVTile + vBlock * 16, qStride);
				mma_sync(dAtt_acc, dOut_frags[vBlock], v_frag, dAtt_acc);
			}
		}

		// Store dAtt to shared memory
		float* myDAttTile = dAttTiles + warpId * tileStride * 16;
		store_matrix_sync(myDAttTile, dAtt_acc, tileStride, wmma::mem_row_major);
		__syncwarp();

		// Load attention weights
		__half* myAttTile = attTiles + warpId * tileStride * 16;
#pragma unroll 4
		for(int idx = laneId; idx < 16 * 16; idx += 32){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = rowStart + r;
			const int globalCol = keyBase + c;
			if(globalRow < tokens && globalCol < tokens){
				const size_t attIdx = attentionOffset + globalRow * tokens + globalCol;
				myAttTile[r * tileStride + c] = Att[attIdx];
			} else{
				myAttTile[r * tileStride + c] = __float2half(0.0f);
			}
		}
		__syncwarp();

		// Accumulate row sums: sum += dAtt * Att element-wise
#pragma unroll 2
		for(int row = 0; row < 16; row++){
			const int globalRow = rowStart + row;
			if(globalRow >= tokens) continue;

			float accum = 0.0f;
#pragma unroll 4
			for(int col = laneId; col < 16; col += 32){
				const int globalCol = keyBase + col;
				if(globalCol < tokens){
					float datt_val = myDAttTile[row * tileStride + col];
					float att_val = __half2float(myAttTile[row * tileStride + col]);
					accum += datt_val * att_val;
				}
			}

			// Warp reduce
#pragma unroll
			for(int offset = 16; offset > 0; offset /= 2){
				accum += __shfl_xor_sync(0xffffffff, accum, offset);
			}

			if(laneId == 0){
				atomicAdd(&rowSums[row], accum);
			}
		}
		__syncwarp();
	}
	__syncthreads();

	// ====== PASS 2: Apply softmax gradient and write to global ======
	for(int keyBlock = warpId; keyBlock < numKeyBlocks; keyBlock += numWarps){
		const int keyBase = keyBlock * 16;
		if(keyBase >= tokens) continue;

		// Recompute dAtt tile
		__half* myVTile = vTiles + warpId * 16 * qStride;
		for(int row = 0; row < 16; ++row){
			const int globalKey = keyBase + row;
			__half* sharedRow = myVTile + row * qStride;
			if(globalKey < tokens){
				const size_t vOffset = batchHeadOffset + globalKey * headDim;
#pragma unroll 4
				for(int col = laneId; col < headDim; col += 32){
					sharedRow[col] = V[vOffset + col];
				}
			}
			for(int col = laneId + headDim; col < qStride; col += 32){
				sharedRow[col] = __float2half(0.0f);
			}
		}
		__syncwarp();

		wmma::fragment<wmma::accumulator, 16, 16, 16, float> dAtt_acc;
		fill_fragment(dAtt_acc, 0.0f);

#pragma unroll
		for(int vBlock = 0; vBlock < vBlocks; ++vBlock){
			if(vBlock * 16 < headDim){
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> v_frag;
				load_matrix_sync(v_frag, myVTile + vBlock * 16, qStride);
				mma_sync(dAtt_acc, dOut_frags[vBlock], v_frag, dAtt_acc);
			}
		}

		float* myDAttTile = dAttTiles + warpId * tileStride * 16;
		store_matrix_sync(myDAttTile, dAtt_acc, tileStride, wmma::mem_row_major);
		__syncwarp();

		// Load attention weights again
		__half* myAttTile = attTiles + warpId * tileStride * 16;
#pragma unroll 4
		for(int idx = laneId; idx < 16 * 16; idx += 32){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = rowStart + r;
			const int globalCol = keyBase + c;
			if(globalRow < tokens && globalCol < tokens){
				const size_t attIdx = attentionOffset + globalRow * tokens + globalCol;
				myAttTile[r * tileStride + c] = Att[attIdx];
			} else{
				myAttTile[r * tileStride + c] = __float2half(0.0f);
			}
		}
		__syncwarp();

		// Apply softmax gradient and write to global: dAtt_final = att * (dAtt - rowSum)
#pragma unroll 4
		for(int idx = laneId; idx < 16 * 16; idx += 32){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = rowStart + r;
			const int globalCol = keyBase + c;
			if(globalRow < tokens && globalCol < tokens){
				float datt_val = myDAttTile[r * tileStride + c];
				float att_val = __half2float(myAttTile[r * tileStride + c]);
				float gradient = att_val * (datt_val - rowSums[r]);

				const size_t attIdx = attentionOffset + globalRow * tokens + globalCol;
				dAttWorkspace[attIdx] = gradient;

				// Store back for dQ computation
				myDAttTile[r * tileStride + c] = gradient;
			}
		}
		__syncwarp();
	}
	__syncthreads();

	// ====== PASS 3: Compute dQ = scale * (dAtt @ K) ======
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_acc[kMaxValueBlocks];
#pragma unroll
	for(int dBlock = 0; dBlock < vBlocks; ++dBlock){
		fill_fragment(dQ_acc[dBlock], 0.0f);
	}

	for(int keyBlock = warpId; keyBlock < numKeyBlocks; keyBlock += numWarps){
		const int keyBase = keyBlock * 16;
		if(keyBase >= tokens) continue;

		// Load K tile
		__half* myKTile = kTiles + warpId * 16 * qStride;
		for(int row = 0; row < 16; ++row){
			const int globalKey = keyBase + row;
			__half* sharedRow = myKTile + row * qStride;
			if(globalKey < tokens){
				const size_t kOffset = batchHeadOffset + globalKey * headDim;
#pragma unroll 4
				for(int col = laneId; col < headDim; col += 32){
					sharedRow[col] = K[kOffset + col];
				}
			}
			for(int col = laneId + headDim; col < qStride; col += 32){
				sharedRow[col] = __float2half(0.0f);
			}
		}
		__syncwarp();

		// Load dAtt from global memory
		float* myDAttTile = dAttTiles + warpId * tileStride * 16;
#pragma unroll 4
		for(int idx = laneId; idx < 16 * 16; idx += 32){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = rowStart + r;
			const int globalCol = keyBase + c;
			float val = 0.0f;
			if(globalRow < tokens && globalCol < tokens){
				const size_t attIdx = attentionOffset + globalRow * tokens + globalCol;
				val = dAttWorkspace[attIdx];
			}
			myDAttTile[r * tileStride + c] = val;
		}
		__syncwarp();

		// Convert to half for WMMA
		__half* myAttTile = attTiles + warpId * tileStride * 16;
#pragma unroll 4
		for(int idx = laneId; idx < 16 * 16; idx += 32){
			myAttTile[idx] = __float2half(myDAttTile[idx]);
		}
		__syncwarp();

		// Load fragments and compute
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> dAtt_frag;
		load_matrix_sync(dAtt_frag, myAttTile, tileStride);

#pragma unroll
		for(int kBlock = 0; kBlock < vBlocks; ++kBlock){
			if(kBlock * 16 < headDim){
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> k_frag;
				load_matrix_sync(k_frag, myKTile + kBlock * 16, qStride);
				mma_sync(dQ_acc[kBlock], dAtt_frag, k_frag, dQ_acc[kBlock]);
			}
		}
		__syncwarp();
	}

	// Store dQ accumulators to shared memory for reduction
#pragma unroll
	for(int dBlock = 0; dBlock < vBlocks; ++dBlock){
		if(dBlock * 16 < headDim){
			store_matrix_sync(dQWorkspace + warpId * vBlocks * 16 * 16 + dBlock * 16 * 16,
							  dQ_acc[dBlock], 16, wmma::mem_row_major);
		}
	}
	__syncthreads();

	// Reduce across warps and write dQ
	for(int row = 0; row < 16; ++row){
		const int globalRow = rowStart + row;
		if(globalRow < tokens){
			const size_t dQRowOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
			for(int col = threadIdx.x; col < headDim; col += blockDim.x){
				const int dBlock = col / 16;
				const int localCol = col % 16;
				const int idx = dBlock * 16 * 16 + row * 16 + localCol;

				float sum = 0.0f;
#pragma unroll
				for(int w = 0; w < numWarps; ++w){
					sum += dQWorkspace[w * vBlocks * 16 * 16 + idx];
				}

				dQ[dQRowOffset + col] = __float2half(sum * scale);
			}
		}
	}
}
// ============================================================================
// OPTIMIZED DV KERNEL WITH WARP-LEVEL PARALLELISM
// ============================================================================
__global__ void __launch_bounds__(256, 2) ComputeDVOptimized(
	const __half* __restrict__ Att, const __half* __restrict__ dOut,
	__half* __restrict__ dV, int batchSize, int tokens, int headDim, int heads){

	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int colBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || colBlock * 16 >= tokens) return;

	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	const int keyStart = colBlock * 16;

	const size_t batchHeadOffset = (static_cast<size_t>(batch) * heads + head) * tokens * headDim;
	const size_t attentionOffset = (static_cast<size_t>(batch) * heads + head) * tokens * tokens;

	const int vBlocks = (headDim + 15) / 16;
	const int numRowBlocks = (tokens + 15) / 16;
	if(vBlocks > kMaxValueBlocks) return;

	extern __shared__ __align__(16) unsigned char sharedStorage[];
	unsigned char* sharedMemBytes = sharedStorage;

	const int tileStride = AlignUp(16 + kSharedMemPad, kSharedMemAlignment);
	const int dOutStride = AlignUp(headDim + kSharedMemPad, kSharedMemAlignment);

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto attTiles = reinterpret_cast<__half*>(sharedMemBytes);
	sharedMemBytes += sizeof(__half) * numWarps * tileStride * 16;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto dOutTiles = reinterpret_cast<__half*>(sharedMemBytes);
	sharedMemBytes += sizeof(__half) * numWarps * 16 * dOutStride;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto floatWorkspace = reinterpret_cast<float*>(sharedMemBytes);

	// Initialize accumulator for this warp
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_acc[kMaxValueBlocks];
#pragma unroll
	for(int dBlock = 0; dBlock < vBlocks; ++dBlock){
		fill_fragment(dV_acc[dBlock], 0.0f);
	}

	// Process tiles with warp-level parallelism
	for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
		const int queryBase = rowBlock * 16;
		if(queryBase >= tokens) continue;

		// Load attention tile (transposed for column-major) for this warp
		__half* myAttTile = attTiles + warpId * tileStride * 16;
#pragma unroll 4
		for(int idx = laneId; idx < 16 * 16; idx += 32){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalQuery = queryBase + r;
			const int globalKey = keyStart + c;
			if(globalQuery < tokens && globalKey < tokens){
				const size_t attIdx = attentionOffset + globalQuery * tokens + globalKey;
				myAttTile[c * tileStride + r] = Att[attIdx];
			} else{
				myAttTile[c * tileStride + r] = __float2half(0.0f);
			}
		}
		__syncwarp();

		// Load dOut tile for this warp
		__half* myDOutTile = dOutTiles + warpId * 16 * dOutStride;
		for(int row = 0; row < 16; ++row){
			const int globalQuery = queryBase + row;
			__half* sharedRow = myDOutTile + row * dOutStride;
			if(globalQuery < tokens){
				const size_t dOutOffset = batchHeadOffset + globalQuery * headDim;
#pragma unroll 4
				for(int col = laneId; col < headDim; col += 32){
					sharedRow[col] = dOut[dOutOffset + col];
				}
			}
			for(int col = laneId + headDim; col < dOutStride; col += 32){
				sharedRow[col] = __float2half(0.0f);
			}
		}
		__syncwarp();

		// Load attention fragment
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> att_frag;
		load_matrix_sync(att_frag, myAttTile, tileStride);

		// Compute dV += Att^T @ dOut
#pragma unroll
		for(int dBlock = 0; dBlock < vBlocks; ++dBlock){
			if(dBlock * 16 < headDim){
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> dOut_frag;
				load_matrix_sync(dOut_frag, myDOutTile + dBlock * 16, dOutStride);
				mma_sync(dV_acc[dBlock], att_frag, dOut_frag, dV_acc[dBlock]);
			}
		}
		__syncwarp();
	}

	// Store accumulators to shared memory for reduction
#pragma unroll
	for(int dBlock = 0; dBlock < vBlocks; ++dBlock){
		if(dBlock * 16 < headDim){
			store_matrix_sync(floatWorkspace + warpId * vBlocks * 16 * 16 + dBlock * 16 * 16,
							  dV_acc[dBlock], 16, wmma::mem_row_major);
		}
	}
	__syncthreads();

	// Reduce across warps and write dV
	for(int row = 0; row < 16; ++row){
		const int globalRow = keyStart + row;
		if(globalRow < tokens){
			const size_t dVOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
			for(int col = threadIdx.x; col < headDim; col += blockDim.x){
				const int dBlock = col / 16;
				const int localCol = col % 16;
				const int idx = dBlock * 16 * 16 + row * 16 + localCol;

				float sum = 0.0f;
#pragma unroll
				for(int w = 0; w < numWarps; ++w){
					sum += floatWorkspace[w * vBlocks * 16 * 16 + idx];
				}

				dV[dVOffset + col] = __float2half(sum);
			}
		}
	}
}
// ============================================================================
// OPTIMIZED DK KERNEL WITH WARP-LEVEL PARALLELISM
// ============================================================================
__global__ void __launch_bounds__(256, 2) ComputeDKOptimized(
	const float* __restrict__ dAtt, const __half* __restrict__ Q,
	__half* __restrict__ dK, int batchSize, int tokens, int headDim, int heads){

	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int colBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || colBlock * 16 >= tokens) return;

	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	const int keyStart = colBlock * 16;

	const size_t batchHeadOffset = (static_cast<size_t>(batch) * heads + head) * tokens * headDim;
	const size_t attentionOffset = (static_cast<size_t>(batch) * heads + head) * tokens * tokens;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));

	const int qBlocks = (headDim + 15) / 16;
	const int numRowBlocks = (tokens + 15) / 16;
	if(qBlocks > kMaxValueBlocks) return;

	extern __shared__ __align__(16) unsigned char sharedStorage[];
	unsigned char* sharedMemBytes = sharedStorage;

	const int tileStride = AlignUp(16 + kSharedMemPad, kSharedMemAlignment);
	const int qStride = AlignUp(headDim + kSharedMemPad, kSharedMemAlignment);

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto dAttTiles = reinterpret_cast<__half*>(sharedMemBytes);
	sharedMemBytes += sizeof(__half) * numWarps * tileStride * 16;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto qTiles = reinterpret_cast<__half*>(sharedMemBytes);
	sharedMemBytes += sizeof(__half) * numWarps * 16 * qStride;

	sharedMemBytes = AlignSharedPtr(sharedMemBytes, kSharedMemAlignment);
	auto floatWorkspace = reinterpret_cast<float*>(sharedMemBytes);

	// Initialize accumulator for this warp
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_acc[kMaxValueBlocks];
#pragma unroll
	for(int qBlock = 0; qBlock < qBlocks; ++qBlock){
		fill_fragment(dK_acc[qBlock], 0.0f);
	}

	// Process tiles with warp-level parallelism
	for(int rowBlock = warpId; rowBlock < numRowBlocks; rowBlock += numWarps){
		const int queryBase = rowBlock * 16;
		if(queryBase >= tokens) continue;

		// Load dAtt tile (transposed for column-major) for this warp
		__half* myDAttTile = dAttTiles + warpId * tileStride * 16;
#pragma unroll 4
		for(int idx = laneId; idx < 16 * 16; idx += 32){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalQuery = queryBase + r;
			const int globalKey = keyStart + c;
			if(globalQuery < tokens && globalKey < tokens){
				const size_t attIdx = attentionOffset + globalQuery * tokens + globalKey;
				// Transpose for column-major: swap r and c
				myDAttTile[c * tileStride + r] = __float2half(dAtt[attIdx]);
			} else{
				myDAttTile[c * tileStride + r] = __float2half(0.0f);
			}
		}
		__syncwarp();

		// Load Q tile for this warp (with scaling)
		__half* myQTile = qTiles + warpId * 16 * qStride;
		for(int row = 0; row < 16; ++row){
			const int globalQuery = queryBase + row;
			__half* sharedRow = myQTile + row * qStride;
			if(globalQuery < tokens){
				const size_t qOffset = batchHeadOffset + globalQuery * headDim;
#pragma unroll 4
				for(int col = laneId; col < headDim; col += 32){
					sharedRow[col] = __hmul(Q[qOffset + col], __float2half(scale));
				}
			}
			for(int col = laneId + headDim; col < qStride; col += 32){
				sharedRow[col] = __float2half(0.0f);
			}
		}
		__syncwarp();

		// Load dAtt fragment
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> dAtt_frag;
		load_matrix_sync(dAtt_frag, myDAttTile, tileStride);

		// Compute dK += dAtt^T @ Q
#pragma unroll
		for(int qBlock = 0; qBlock < qBlocks; ++qBlock){
			if(qBlock * 16 < headDim){
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> q_frag;
				load_matrix_sync(q_frag, myQTile + qBlock * 16, qStride);
				mma_sync(dK_acc[qBlock], dAtt_frag, q_frag, dK_acc[qBlock]);
			}
		}
		__syncwarp();
	}

	// Store accumulators to shared memory for reduction
#pragma unroll
	for(int qBlock = 0; qBlock < qBlocks; ++qBlock){
		if(qBlock * 16 < headDim){
			store_matrix_sync(floatWorkspace + warpId * qBlocks * 16 * 16 + qBlock * 16 * 16,
							  dK_acc[qBlock], 16, wmma::mem_row_major);
		}
	}
	__syncthreads();

	// Reduce across warps and write dK
	for(int row = 0; row < 16; ++row){
		const int globalRow = keyStart + row;
		if(globalRow < tokens){
			const size_t dKOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
			for(int col = threadIdx.x; col < headDim; col += blockDim.x){
				const int qBlock = col / 16;
				const int localCol = col % 16;
				const int idx = qBlock * 16 * 16 + row * 16 + localCol;

				float sum = 0.0f;
#pragma unroll
				for(int w = 0; w < numWarps; ++w){
					sum += floatWorkspace[w * qBlocks * 16 * 16 + idx];
				}

				dK[dKOffset + col] = __float2half(sum);
			}
		}
	}
}
// ============================================================================
// OPTIMIZED BACKWARD ENTRY POINT
// ============================================================================
void WmmaAttentionBackwardv2(const __half* Q, const __half* K, const __half* V, const __half* dOut, const __half* Att, __half* dQ, __half* dK, __half* dV, float* dAttWorkspace, size_t workspaceElements, int batchSize, int tokens, int headDim, int heads){
	// Input validation
	if(!Q || !K || !V || !dOut || !Att || !dQ || !dK || !dV || !dAttWorkspace){
		printf("WmmaAttentionBackwardv2: Null pointer(s) provided\n");
		return;
	}
	const size_t requiredElements = static_cast<size_t>(batchSize) * heads * tokens * tokens;
	if(requiredElements > workspaceElements){
		printf("WmmaAttentionBackwardv2: Workspace too small (need %zu, have %zu)\n", requiredElements, workspaceElements);
		return;
	}

	const int numBlocks = DivCeil(tokens, 16);
	const int vBlocks = (headDim + 15) / 16;

	if(vBlocks > kMaxValueBlocks){
		printf("WmmaAttentionBackwardv2: headDim %d requires %d blocks, exceeds limit %d\n", headDim, vBlocks, kMaxValueBlocks);
		return;
	}

	// Use 256 threads (8 warps) for better parallelism
	constexpr int kThreads = 256;
	constexpr int kNumWarps = kThreads / 32;

	dim3 block(kThreads);
	dim3 grid(numBlocks, batchSize, heads);

	const int tileStride = AlignUp(16 + kSharedMemPad, kSharedMemAlignment);
	const int qStride = AlignUp(headDim + kSharedMemPad, kSharedMemAlignment);
	const int dOutStride = AlignUp(headDim + kSharedMemPad, kSharedMemAlignment);

	const auto alignBytes = [](size_t offset){ return AlignUp(offset, static_cast<size_t>(kSharedMemAlignment)); };

	// Calculate shared memory for ComputeDAttDQOptimized
	size_t smemDQ = 0;
	smemDQ = alignBytes(smemDQ);
	smemDQ += sizeof(__half) * 16 * qStride; // dOutShared
	smemDQ = alignBytes(smemDQ);
	smemDQ += sizeof(__half) * kNumWarps * 16 * qStride; // vTiles
	smemDQ = alignBytes(smemDQ);
	smemDQ += sizeof(__half) * kNumWarps * 16 * qStride; // kTiles
	smemDQ = alignBytes(smemDQ);
	smemDQ += sizeof(float) * kNumWarps * tileStride * 16; // dAttTiles
	smemDQ = alignBytes(smemDQ);
	smemDQ += sizeof(__half) * kNumWarps * tileStride * 16; // attTiles
	smemDQ = alignBytes(smemDQ);
	smemDQ += sizeof(float) * 16; // rowSums
	smemDQ = alignBytes(smemDQ);
	smemDQ += sizeof(float) * kNumWarps * vBlocks * 16 * 16; // dQWorkspace

	// Calculate shared memory for ComputeDVOptimized
	size_t smemDV = 0;
	smemDV = alignBytes(smemDV);
	smemDV += sizeof(__half) * kNumWarps * tileStride * 16; // attTiles
	smemDV = alignBytes(smemDV);
	smemDV += sizeof(__half) * kNumWarps * 16 * dOutStride; // dOutTiles
	smemDV = alignBytes(smemDV);
	smemDV += sizeof(float) * kNumWarps * vBlocks * 16 * 16; // floatWorkspace

	// Calculate shared memory for ComputeDKOptimized
	size_t smemDK = 0;
	smemDK = alignBytes(smemDK);
	smemDK += sizeof(__half) * kNumWarps * tileStride * 16; // dAttTiles
	smemDK = alignBytes(smemDK);
	smemDK += sizeof(__half) * kNumWarps * 16 * qStride; // qTiles
	smemDK = alignBytes(smemDK);
	smemDK += sizeof(float) * kNumWarps * vBlocks * 16 * 16; // floatWorkspace

	// Check if shared memory requirements are within limits
	if(smemDQ > kVoltaSharedMemory || smemDV > kVoltaSharedMemory || smemDK > kVoltaSharedMemory){
		printf("WmmaAttentionBackwardv2: Shared memory requirements exceed limits\n");
		printf("  dQ kernel: %zu bytes (max %d)\n", smemDQ, kVoltaSharedMemory);
		printf("  dV kernel: %zu bytes (max %d)\n", smemDV, kVoltaSharedMemory);
		printf("  dK kernel: %zu bytes (max %d)\n", smemDK, kVoltaSharedMemory);
		return;
	}

	// Set shared memory configuration
	cudaFuncSetAttribute(ComputeDAttDQOptimized, cudaFuncAttributeMaxDynamicSharedMemorySize, kVoltaSharedMemory);
	cudaFuncSetAttribute(ComputeDVOptimized, cudaFuncAttributeMaxDynamicSharedMemorySize, kVoltaSharedMemory);
	cudaFuncSetAttribute(ComputeDKOptimized, cudaFuncAttributeMaxDynamicSharedMemorySize, kVoltaSharedMemory);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	// Launch optimized kernels
	ComputeDAttDQOptimized<<<grid, block, smemDQ>>>(Q, K, V, dOut, Att, dQ, dAttWorkspace, batchSize, tokens, headDim, heads);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttentionBackwardv2 dQ kernel error: %s\n", cudaGetErrorString(err));
		return;
	}

	ComputeDVOptimized<<<grid, block, smemDV>>>(Att, dOut, dV, batchSize, tokens, headDim, heads);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttentionBackwardv2 dV kernel error: %s\n", cudaGetErrorString(err));
		return;
	}

	ComputeDKOptimized<<<grid, block, smemDK>>>(dAttWorkspace, Q, dK, batchSize, tokens, headDim, heads);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttentionBackwardv2 dK kernel error: %s\n", cudaGetErrorString(err));
	}
}