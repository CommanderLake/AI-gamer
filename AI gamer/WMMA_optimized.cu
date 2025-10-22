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
	__host__ __device__ inline int DivCeil(int a, int b){ return (a + b - 1) / b; }
	// Helper function to convert float fragment to half fragment
	template <typename FragmentType> __device__ inline void ConvertFloatFragmentToHalf(FragmentType& float_frag, wmma::fragment<wmma::accumulator, 16, 16, 16, __half>& half_frag){
#pragma unroll
		for(int i = 0; i < float_frag.num_elements; ++i){ half_frag.x[i] = __float2half(float_frag.x[i]); }
	}
}
// ============================================================================
// OPTIMIZED BACKWARD KERNEL WITH PROPER TYPE HANDLING
// ============================================================================
__global__ void __launch_bounds__(128, 2) FusedBackwardKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const __half* __restrict__ Att, __half* __restrict__ dQ, __half* __restrict__ dK,
																	float* __restrict__ dAttWorkspace, int batchSize, int tokens, int headDim, int heads){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || rowBlock * 16 >= tokens) return;
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	extern __shared__ char sharedMemBytes[];
	const int qStride = ((headDim + 15) / 16 * 16) + kSharedMemPad;
	const int tileStride = 16 + kSharedMemPad;
	const int vBlocks = (headDim + 15) / 16;
	// Shared memory layout with proper float workspace
	auto dOutShared = reinterpret_cast<__half*>(sharedMemBytes);
	auto vShared = dOutShared + 16 * qStride;
	auto kShared = vShared + 16 * qStride;
	auto attShared = kShared + 16 * qStride;
	auto floatWorkspace = reinterpret_cast<float*>(attShared + tileStride * 16);
	auto dAttTile = floatWorkspace;
	auto dQFloat = dAttTile + 16 * 16;
	const size_t batchHeadOffset = (static_cast<size_t>(batch) * heads + head) * tokens * headDim;
	const size_t attentionOffset = (static_cast<size_t>(batch) * heads + head) * tokens * tokens;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	// WMMA fragments
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> dOut_frags[kMaxValueBlocks];
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> v_frag;
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> att_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> dAtt_acc;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> dQ_acc[kMaxValueBlocks];
	// Load dOut to shared memory
	for(int row = 0; row < 16; ++row){
		const int globalRow = rowBlock * 16 + row;
		__half* sharedRow = dOutShared + row * qStride;
		if(globalRow < tokens){
			const size_t dOutRowOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
			for(int col = threadIdx.x; col < headDim; col += blockDim.x){ sharedRow[col] = dOut[dOutRowOffset + col]; }
		} else{ for(int col = threadIdx.x; col < qStride; col += blockDim.x){ sharedRow[col] = __float2half(0.0f); } }
	}
	__syncthreads();
	// Load dOut fragments
#pragma unroll
	for(int dBlock = 0; dBlock < vBlocks; ++dBlock){ if(dBlock * 16 < headDim){ load_matrix_sync(dOut_frags[dBlock], dOutShared + dBlock * 16, qStride); } }
	// Initialize dQ accumulators
#pragma unroll
	for(int dBlock = 0; dBlock < vBlocks; ++dBlock){ fill_fragment(dQ_acc[dBlock], 0.0f); }
	// Process tiles
	for(int tileCol = 0; tileCol < tokens; tileCol += 16){
		if(tileCol >= tokens) break;
		// Compute dAtt = dOut @ V^T
		fill_fragment(dAtt_acc, 0.0f);
		// Load V tile
		for(int row = threadIdx.x / 16; row < 16; row += numWarps * 2){
			const int col = (threadIdx.x % 16);
			const int globalCol = tileCol + row;
			if(globalCol < tokens){
				const size_t vOffset = batchHeadOffset + globalCol * headDim;
#pragma unroll 4
				for(int d = col; d < headDim; d += 16){ vShared[row * qStride + d] = V[vOffset + d]; }
			}
		}
		__syncthreads();
		// Compute dAtt using tensor cores
#pragma unroll
		for(int vBlock = 0; vBlock < vBlocks; ++vBlock){
			if(vBlock * 16 < headDim){
				load_matrix_sync(v_frag, vShared + vBlock * 16, qStride);
				mma_sync(dAtt_acc, dOut_frags[vBlock], v_frag, dAtt_acc);
			}
		}
		// Scale dAtt
#pragma unroll
		for(int i = 0; i < dAtt_acc.num_elements; ++i){ dAtt_acc.x[i] *= scale; }
		// Store dAtt to float workspace
		store_matrix_sync(dAttTile, dAtt_acc, 16, wmma::mem_row_major);
		__syncthreads();
		// Write dAtt to global memory
		const int globalRowStart = rowBlock * 16;
		const int globalColStart = tileCol;
#pragma unroll 4
		for(int i = threadIdx.x; i < 16 * 16; i += blockDim.x){
			const int localRow = i / 16;
			const int localCol = i % 16;
			const int globalRow = globalRowStart + localRow;
			const int globalCol = globalColStart + localCol;
			if(globalRow < tokens && globalCol < tokens){
				const size_t idx = attentionOffset + globalRow * tokens + globalCol;
				dAttWorkspace[idx] = dAttTile[localRow * 16 + localCol];
			}
		}
		__syncthreads();
		// Load attention weights
		for(int row = threadIdx.x / 16; row < 16; row += numWarps * 2){
			const int col = (threadIdx.x % 16);
			const int globalRow = rowBlock * 16 + row;
			const int globalCol = tileCol + col;
			if(globalRow < tokens && globalCol < tokens){
				const size_t idx = attentionOffset + globalRow * tokens + globalCol;
				attShared[row * tileStride + col] = Att[idx];
			} else{ attShared[row * tileStride + col] = __float2half(0.0f); }
		}
		__syncthreads();
		// Load K tile for dQ computation
		for(int row = threadIdx.x / 16; row < 16; row += numWarps * 2){
			const int col = (threadIdx.x % 16);
			const int globalCol = tileCol + row;
			if(globalCol < tokens){
				const size_t kOffset = batchHeadOffset + globalCol * headDim;
#pragma unroll 4
				for(int d = col; d < headDim; d += 16){ kShared[row * qStride + d] = __hmul(K[kOffset + d], __float2half(scale)); }
			}
		}
		__syncthreads();
		// Compute dQ contribution
		load_matrix_sync(att_frag, attShared, tileStride);
		// Apply dAtt scaling to attention weights
#pragma unroll
		for(int i = 0; i < att_frag.num_elements; ++i){
			float att_val = __half2float(att_frag.x[i]);
			float datt_val = dAttTile[i];
			att_frag.x[i] = __float2half(att_val * datt_val);
		}
#pragma unroll
		for(int kBlock = 0; kBlock < vBlocks; ++kBlock){
			if(kBlock * 16 < headDim){
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> k_frag;
				load_matrix_sync(k_frag, kShared + kBlock * 16, qStride);
				mma_sync(dQ_acc[kBlock], att_frag, k_frag, dQ_acc[kBlock]);
			}
		}
	}
	// Write dQ to global memory
	const int globalRowStart = rowBlock * 16;
	// Store float accumulators to float workspace first
#pragma unroll
	for(int dBlock = 0; dBlock < vBlocks; ++dBlock){ if(dBlock * 16 < headDim){ store_matrix_sync(dQFloat + dBlock * 16 * 16, dQ_acc[dBlock], 16, wmma::mem_row_major); } }
	__syncthreads();
	// Convert float to half and reorganize
#pragma unroll 4
	for(int i = threadIdx.x; i < vBlocks * 16 * 16; i += blockDim.x){
		const int block = i / (16 * 16);
		const int idx = i % (16 * 16);
		const int row = idx / 16;
		const int col = idx % 16;
		if(block * 16 + col < headDim){ dOutShared[row * qStride + block * 16 + col] = __float2half(dQFloat[i]); }
	}
	__syncthreads();
	// Write to global dQ
	for(int row = 0; row < 16; ++row){
		const int globalRow = globalRowStart + row;
		if(globalRow < tokens){
			const size_t dQRowOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
			for(int col = threadIdx.x; col < headDim; col += blockDim.x){ dQ[dQRowOffset + col] = dOutShared[row * qStride + col]; }
		}
	}
}
// ============================================================================
// OPTIMIZED DV KERNEL WITH PROPER TYPE HANDLING
// ============================================================================
__global__ void __launch_bounds__(128, 2) ComputeDV(const __half* __restrict__ Att, const __half* __restrict__ dOut, __half* __restrict__ dV, int batchSize, int tokens, int headDim, int heads){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int colBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || colBlock * 16 >= tokens) return;
	extern __shared__ char sharedMemBytes[];
	const int vBlocks = (headDim + 15) / 16;
	const int tileStride = 16 + kSharedMemPad;
	const int dOutStride = ((headDim + 15) / 16 * 16) + kSharedMemPad;
	auto attTile = reinterpret_cast<__half*>(sharedMemBytes);
	auto dOutTiles = attTile + tileStride * 16;
	auto floatWorkspace = reinterpret_cast<float*>(dOutTiles + 16 * dOutStride);
	const size_t batchHeadOffset = (static_cast<size_t>(batch) * heads + head) * tokens * headDim;
	const size_t attentionOffset = (static_cast<size_t>(batch) * heads + head) * tokens * tokens;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> dV_acc[kMaxValueBlocks];
	// Initialize accumulators
#pragma unroll
	for(int i = 0; i < vBlocks; ++i){ fill_fragment(dV_acc[i], 0.0f); }
	// Process tiles
	for(int tileRow = 0; tileRow < tokens; tileRow += 16){
		if(tileRow >= tokens) break;
		// Load attention tile (transposed for column-major)
#pragma unroll 4
		for(int i = threadIdx.x; i < 16 * 16; i += blockDim.x){
			const int localRow = i / 16;
			const int localCol = i % 16;
			const int globalRow = tileRow + localRow;
			const int globalCol = colBlock * 16 + localCol;
			if(globalRow < tokens && globalCol < tokens){ attTile[localCol * tileStride + localRow] = Att[attentionOffset + globalRow * tokens + globalCol]; } else{ attTile[localCol * tileStride + localRow] = __float2half(0.0f); }
		}
		// Load dOut tiles
		for(int row = 0; row < 16; ++row){
			const int globalRow = tileRow + row;
			if(globalRow < tokens){
				const size_t dOutOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
				for(int col = threadIdx.x; col < headDim; col += blockDim.x){ dOutTiles[row * dOutStride + col] = dOut[dOutOffset + col]; }
			} else{ for(int col = threadIdx.x; col < dOutStride; col += blockDim.x){ dOutTiles[row * dOutStride + col] = __float2half(0.0f); } }
		}
		__syncthreads();
		// Compute dV += Att^T @ dOut using tensor cores
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> att_frag;
		load_matrix_sync(att_frag, attTile, tileStride);
#pragma unroll
		for(int dBlock = 0; dBlock < vBlocks; ++dBlock){
			if(dBlock * 16 < headDim){
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> dOut_frag;
				load_matrix_sync(dOut_frag, dOutTiles + dBlock * 16, dOutStride);
				mma_sync(dV_acc[dBlock], att_frag, dOut_frag, dV_acc[dBlock]);
			}
		}
		__syncthreads();
	}
	// Write results - first to float workspace
	const int globalCol = colBlock * 16;
#pragma unroll
	for(int dBlock = 0; dBlock < vBlocks; ++dBlock){ if(dBlock * 16 < headDim){ store_matrix_sync(floatWorkspace + dBlock * 16 * 16, dV_acc[dBlock], 16, wmma::mem_row_major); } }
	__syncthreads();
	// Convert float to half
#pragma unroll 4
	for(int i = threadIdx.x; i < vBlocks * 16 * 16; i += blockDim.x){
		const int block = i / (16 * 16);
		const int idx = i % (16 * 16);
		const int row = idx / 16;
		const int col = idx % 16;
		if(block * 16 + col < headDim){ dOutTiles[row * dOutStride + block * 16 + col] = __float2half(floatWorkspace[i]); }
	}
	__syncthreads();
	// Write to global dV with atomic adds
	for(int row = 0; row < 16; ++row){
		const int globalRow = globalCol + row;
		if(globalRow < tokens){
			const size_t dVOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
			for(int col = threadIdx.x; col < headDim; col += blockDim.x){
				__half val = dOutTiles[row * dOutStride + col];
				// Use atomic add for accumulation
				atomicAdd(&dV[dVOffset + col], val);
			}
		}
	}
}
// ============================================================================
// OPTIMIZED DK KERNEL WITH PROPER TYPE HANDLING  
// ============================================================================
__global__ void __launch_bounds__(128, 2) ComputeDK(const float* __restrict__ dAtt, const __half* __restrict__ Q, __half* __restrict__ dK, int batchSize, int tokens, int headDim, int heads){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int colBlock = blockIdx.x;
	if(batch >= batchSize || head >= heads || colBlock * 16 >= tokens) return;
	extern __shared__ char sharedMemBytes[];
	const int qBlocks = (headDim + 15) / 16;
	const int tileStride = 16 + kSharedMemPad;
	const int qStride = ((headDim + 15) / 16 * 16) + kSharedMemPad;
	auto dAttTileHalf = reinterpret_cast<__half*>(sharedMemBytes);
	auto qTiles = dAttTileHalf + tileStride * 16;
	auto floatWorkspace = reinterpret_cast<float*>(qTiles + 16 * qStride);
	const size_t batchHeadOffset = (static_cast<size_t>(batch) * heads + head) * tokens * headDim;
	const size_t attentionOffset = (static_cast<size_t>(batch) * heads + head) * tokens * tokens;
	const float scale = rsqrtf(fmaxf(static_cast<float>(headDim), 1.0f));
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> dK_acc[kMaxValueBlocks];
	// Initialize accumulators
#pragma unroll
	for(int i = 0; i < qBlocks; ++i){ fill_fragment(dK_acc[i], 0.0f); }
	// Process tiles
	for(int tileRow = 0; tileRow < tokens; tileRow += 16){
		if(tileRow >= tokens) break;
		// Load dAtt tile (transposed) and convert to half
#pragma unroll 4
		for(int i = threadIdx.x; i < 16 * 16; i += blockDim.x){
			const int localRow = i / 16;
			const int localCol = i % 16;
			const int globalRow = tileRow + localRow;
			const int globalCol = colBlock * 16 + localCol;
			if(globalRow < tokens && globalCol < tokens){
				float datt_val = dAtt[attentionOffset + globalRow * tokens + globalCol];
				// Transpose for column-major
				dAttTileHalf[localCol * tileStride + localRow] = __float2half(datt_val);
			} else{ dAttTileHalf[localCol * tileStride + localRow] = __float2half(0.0f); }
		}
		// Load Q tiles
		for(int row = 0; row < 16; ++row){
			const int globalRow = tileRow + row;
			if(globalRow < tokens){
				const size_t qOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
				for(int col = threadIdx.x; col < headDim; col += blockDim.x){ qTiles[row * qStride + col] = __hmul(Q[qOffset + col], __float2half(scale)); }
			} else{ for(int col = threadIdx.x; col < qStride; col += blockDim.x){ qTiles[row * qStride + col] = __float2half(0.0f); } }
		}
		__syncthreads();
		// Load dAtt fragment
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> dAtt_frag;
		load_matrix_sync(dAtt_frag, dAttTileHalf, tileStride);
		// Compute dK += dAtt^T @ Q
#pragma unroll
		for(int qBlock = 0; qBlock < qBlocks; ++qBlock){
			if(qBlock * 16 < headDim){
				wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> q_frag;
				load_matrix_sync(q_frag, qTiles + qBlock * 16, qStride);
				mma_sync(dK_acc[qBlock], dAtt_frag, q_frag, dK_acc[qBlock]);
			}
		}
		__syncthreads();
	}
	// Write results - first to float workspace
	const int globalCol = colBlock * 16;
#pragma unroll
	for(int qBlock = 0; qBlock < qBlocks; ++qBlock){ if(qBlock * 16 < headDim){ store_matrix_sync(floatWorkspace + qBlock * 16 * 16, dK_acc[qBlock], 16, wmma::mem_row_major); } }
	__syncthreads();
	// Convert float to half
#pragma unroll 4
	for(int i = threadIdx.x; i < qBlocks * 16 * 16; i += blockDim.x){
		const int block = i / (16 * 16);
		const int idx = i % (16 * 16);
		const int row = idx / 16;
		const int col = idx % 16;
		if(block * 16 + col < headDim){ qTiles[row * qStride + block * 16 + col] = __float2half(floatWorkspace[i]); }
	}
	__syncthreads();
	// Write to global dK with atomic adds
	for(int row = 0; row < 16; ++row){
		const int globalRow = globalCol + row;
		if(globalRow < tokens){
			const size_t dKOffset = batchHeadOffset + globalRow * headDim;
#pragma unroll 4
			for(int col = threadIdx.x; col < headDim; col += blockDim.x){
				__half val = qTiles[row * qStride + col];
				atomicAdd(&dK[dKOffset + col], val);
			}
		}
	}
}
// ============================================================================
// BACKWARD ENTRY POINT WITH FIXED TYPE HANDLING
// ============================================================================
void WmmaAttentionBackwardv2(const __half* Q, const __half* K, const __half* V, const __half* dOut, const __half* Att, __half* dQ, __half* dK, __half* dV, float* dAttWorkspace, size_t workspaceElements, int batchSize, int tokens, int headDim, int heads){
	// Input validation
	if(!Q || !K || !V || !dOut || !Att || !dQ || !dK || !dV || !dAttWorkspace){
		printf("WmmaAttentionBackward: Null pointer(s) provided\n");
		return;
	}
	const size_t requiredElements = static_cast<size_t>(batchSize) * heads * tokens * tokens;
	if(requiredElements > workspaceElements){
		printf("WmmaAttentionBackward: Workspace too small\n");
		return;
	}
	const int numBlocks = DivCeil(tokens, 16);
	// Clear outputs
	cudaMemset(dQ, 0, batchSize * heads * tokens * headDim * sizeof(__half));
	cudaMemset(dK, 0, batchSize * heads * tokens * headDim * sizeof(__half));
	cudaMemset(dV, 0, batchSize * heads * tokens * headDim * sizeof(__half));
	dim3 block(kOptimalThreadsVolta);
	dim3 grid(numBlocks, batchSize, heads);
	const int tileStride = 16 + kSharedMemPad;
	const int qStride = ((headDim + 15) / 16 * 16) + kSharedMemPad;
	const int vBlocks = (headDim + 15) / 16;
	// Calculate shared memory with proper float workspace
	const size_t smemFused = sizeof(__half) * (4 * 16 * qStride + tileStride * 16) + sizeof(float) * (16 * 16 + vBlocks * 16 * 16);
	const size_t smemDV = sizeof(__half) * (tileStride * 16 + 16 * qStride) + sizeof(float) * (vBlocks * 16 * 16);
	const size_t smemDK = sizeof(__half) * (tileStride * 16 + 16 * qStride) + sizeof(float) * (vBlocks * 16 * 16);
	// Set shared memory configuration
	cudaFuncSetAttribute(FusedBackwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kVoltaSharedMemory);
	cudaFuncSetAttribute(ComputeDV, cudaFuncAttributeMaxDynamicSharedMemorySize, kVoltaSharedMemory);
	cudaFuncSetAttribute(ComputeDK, cudaFuncAttributeMaxDynamicSharedMemorySize, kVoltaSharedMemory);
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
	// Launch kernels
	FusedBackwardKernel<<<grid, block, smemFused>>>(Q, K, V, dOut, Att, dQ, dK, dAttWorkspace, batchSize, tokens, headDim, heads);
	ComputeDV<<<grid, block, smemDV>>>(Att, dOut, dV, batchSize, tokens, headDim, heads);
	ComputeDK<<<grid, block, smemDK>>>(dAttWorkspace, Q, dK, batchSize, tokens, headDim, heads);
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){ printf("WmmaAttentionBackward error: %s\n", cudaGetErrorString(err)); }
}