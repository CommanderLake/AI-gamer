#include "CuCommon.cuh"
#define __CUDACC__
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
__device__ __half AtomicAddHalf(__half* addr, __half val){
	const uintptr_t uptr = reinterpret_cast<uintptr_t>(addr);
	const auto base = reinterpret_cast<unsigned int*>(uptr & ~0x3);
	const bool upper = (uptr & 0x2) != 0;
	unsigned int old = *base, assumed;
	do{
		assumed = old;
		const __half2 pair = *reinterpret_cast<__half2 const*>(&assumed);
		const __half cur = upper ? __high2half(pair) : __low2half(pair);
		const float f = __half2float(cur) + __half2float(val);
		const __half newh = __float2half(f);
		__half2 newpair = upper ? __halves2half2(__low2half(pair), newh) : __halves2half2(newh, __high2half(pair));
		old = atomicCAS(base, assumed, *reinterpret_cast<unsigned int*>(&newpair));
	} while(old != assumed);
	const __half2 finalPair = *reinterpret_cast<__half2*>(&old);
	const __half oldHalf = upper ? __high2half(finalPair) : __low2half(finalPair);
	return oldHalf;
}
__global__ void WmmaAttentionKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ Out, float* __restrict__ AttentionWeights, int B, int T, int D, int H){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int row_block = blockIdx.x;
	if(row_block*16 >= T) return;
	const int warp_id = threadIdx.x / 32;
	const int lane_id = threadIdx.x % 32;
	const int num_warps = blockDim.x / 32;
	const int batch_head_offset = (batch*H + head)*T*D;
	extern __shared__ char shared_mem_bytes[];
	auto Q_shared = (__half*)shared_mem_bytes;
	__half* K_shared = Q_shared + 16*D;
	__half* K_transposed = K_shared + 16*D;
	__half* V_tile = K_transposed + 16*16;
	auto scores_shared = (float*)(V_tile + 16*16);
	float* row_max = scores_shared + 16*T;
	float* row_sum = row_max + 16;
	auto att_tile = (__half*)(row_sum + 16);
	auto warp_row_max = reinterpret_cast<float*>(att_tile + 16*16);
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> k_frag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> v_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> scores_frag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> out_frag;
	if(threadIdx.x < 16){
		row_max[threadIdx.x] = -FLT_MAX;
		row_sum[threadIdx.x] = 0.0f;
	}
#pragma unroll
	for(int d = threadIdx.x; d < D*16; d += blockDim.x){
		const int row = d / D;
		const int col = d % D;
		if(row_block*16 + row < T){ Q_shared[row*D + col] = Q[batch_head_offset + (row_block*16 + row)*D + col]; } else{ Q_shared[row*D + col] = __float2half(0.0f); }
	}
	__syncthreads();
	for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
		fill_fragment(scores_frag, 0.0f);
#pragma unroll
		for(int d = threadIdx.x; d < D*16; d += blockDim.x){
			const int row = d / D;
			const int col = d % D;
			if(col_block*16 + row < T){ K_shared[row*D + col] = K[batch_head_offset + (col_block*16 + row)*D + col]; } else{ K_shared[row*D + col] = __float2half(0.0f); }
		}
		__syncthreads();
#pragma unroll
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			if(warp_id == 0){
#pragma unroll
				for(int i = lane_id; i < 16*16; i += 32){
					const int row = i / 16;
					const int col = i % 16;
					if(d_block*16 + col < D){ K_transposed[col*16 + row] = K_shared[row*D + d_block*16 + col]; } else{ K_transposed[col*16 + row] = __float2half(0.0f); }
				}
			}
			__syncthreads();
			load_matrix_sync(q_frag, Q_shared + d_block*16, D);
			load_matrix_sync(k_frag, K_transposed, 16);
			mma_sync(scores_frag, q_frag, k_frag, scores_frag);
		}
#pragma unroll
		for(int i = 0; i < scores_frag.num_elements; i++){ scores_frag.x[i] *= rsqrtf(static_cast<float>(D)); }
		if(warp_id == 0){ store_matrix_sync(scores_shared + col_block*16, scores_frag, T, wmma::mem_row_major); }
		__syncthreads();
	}
#pragma unroll
	for(int row = 0; row < 16; row++){
		float m = -FLT_MAX;
		for(int col = threadIdx.x; col < T; col += blockDim.x){ if(row_block*16 + row < T && col < T){ m = fmaxf(m, scores_shared[row*T + col]); } }
#pragma unroll
		for(int offset = 16; offset > 0; offset /= 2){ m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset)); }
		if(lane_id == 0){ warp_row_max[row*num_warps + warp_id] = m; }
	}
	__syncthreads();
	if(threadIdx.x < 16){
		float max_val = -FLT_MAX;
		for(int w = 0; w < num_warps; w++){ max_val = fmaxf(max_val, warp_row_max[threadIdx.x*num_warps + w]); }
		row_max[threadIdx.x] = max_val;
	}
	__syncthreads();
#pragma unroll
	for(int i = threadIdx.x; i < 16*T; i += blockDim.x){
		const int row = i / T;
		const int col = i % T;
		if(row_block*16 + row < T && col < T){
			const float val = expf(scores_shared[row*T + col] - row_max[row]);
			scores_shared[row*T + col] = val;
			atomicAdd(&row_sum[row], val);
		}
	}
	__syncthreads();
#pragma unroll
	for(int i = threadIdx.x; i < 16*T; i += blockDim.x){
		const int row = i / T;
		const int col = i % T;
		if(row_block*16 + row < T && col < T){
			const float normalized = scores_shared[row*T + col] / row_sum[row];
			scores_shared[row*T + col] = normalized;
			AttentionWeights[batch*H*T*T + head*T*T + (row_block*16 + row)*T + col] = normalized;
		}
	}
	__syncthreads();
	for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
		fill_fragment(out_frag, 0.0f);
#pragma unroll
		for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> att_frag;
			if(warp_id == 0){
#pragma unroll
				for(int i = lane_id; i < 16*16; i += 32){
					const int row = i / 16;
					const int col = i % 16;
					if(col_block*16 + col < T){ att_tile[row*16 + col] = __float2half(scores_shared[row*T + col_block*16 + col]); } else{ att_tile[row*16 + col] = __float2half(0.0f); }
				}
			}
			__syncthreads();
#pragma unroll
			for(int i = threadIdx.x; i < 16*16; i += blockDim.x){
				const int row = i / 16;
				const int col = i % 16;
				if(col_block*16 + row < T && d_block*16 + col < D){ V_tile[row*16 + col] = V[batch_head_offset + (col_block*16 + row)*D + d_block*16 + col]; } else{ V_tile[row*16 + col] = __float2half(0.0f); }
			}
			__syncthreads();
			load_matrix_sync(att_frag, att_tile, 16);
			load_matrix_sync(v_frag, V_tile, 16);
			mma_sync(out_frag, att_frag, v_frag, out_frag);
		}
		if(warp_id == 0){
#pragma unroll
			for(int i = 0; i < out_frag.num_elements; i++){
				const int row = i / 16;
				const int col = i % 16;
				if(row_block*16 + row < T && d_block*16 + col < D){ Out[batch_head_offset + (row_block*16 + row)*D + d_block*16 + col] = __float2half(out_frag.x[i]); }
			}
		}
		__syncthreads();
	}
}
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, float* AttentionWeights, int B, int T, int D, int H){
	dim3 block(128);
	dim3 grid((T + 15) / 16, B, H);
	const int num_warps = block.x / 32;
	size_t shared_size = sizeof(__half)*(16*D*2 + 16*16 + 16*16 + 16*16) + sizeof(float)*(16*T + 16 + 16 + 16*num_warps);
	cudaFuncSetAttribute(WmmaAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	WmmaAttentionKernel<<<grid, block, shared_size>>>(Q, K, V, Out, AttentionWeights, B, T, D, H);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("WmmaAttention Forward error: %s\n", cudaGetErrorString(e));
}
__global__ void WmmaAttentionBackwardKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ AttentionWeights, __half* __restrict__ dQ, __half* __restrict__ dK, __half* __restrict__ dV, int B, int T,
											int D, int H){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int rowBlock = blockIdx.x;
	if(rowBlock*16 >= T) return;
	const int warpId = threadIdx.x / 32;
	const int laneId = threadIdx.x % 32;
	const int numWarps = blockDim.x / 32;
	const int batchHeadOffset = (batch*H + head)*T*D;
	const int attOffset = (batch*H + head)*T*T;
	extern __shared__ char sharedMemBytes[];
	auto dOutShared = reinterpret_cast<__half*>(sharedMemBytes);
	__half* vShared = dOutShared + 16*(D + 7 & ~7);
	__half* kShared = vShared + 16*(D + 7 & ~7);
	__half* qShared = kShared + 16*(D + 7 & ~7);
	auto attShared = reinterpret_cast<float*>(qShared + 16*(D + 7 & ~7));
	float* dAttShared = attShared + 16*(T + 3 & ~3);
	float* rowsumShared = dAttShared + 16*(T + 3 & ~3);
	float* warpRowsum = rowsumShared + 16;
	auto workspaceHalf = reinterpret_cast<__half*>(warpRowsum + 16*numWarps);
	auto workspaceFloat = reinterpret_cast<float*>(workspaceHalf + 512);
	wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aFrag;
	wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> cFrag;
	const int D_padded = D + 7 & ~7;
	const int T_padded = T + 3 & ~3;
	if(threadIdx.x < 16){ rowsumShared[threadIdx.x] = 0.0f; }
#pragma unroll
	for(int d = threadIdx.x; d < 16*D; d += blockDim.x){
		int row = d / D;
		int col = d % D;
		if(rowBlock*16 + row < T && col < D){ dOutShared[row*D_padded + col] = dOut[batchHeadOffset + (rowBlock*16 + row)*D + col]; } else{ dOutShared[row*D_padded + col] = __float2half(0.0f); }
	}
#pragma unroll
	for(int i = threadIdx.x; i < 16*T; i += blockDim.x){
		int row = i / T;
		int col = i % T;
		if(rowBlock*16 + row < T && col < T){ attShared[row*T_padded + col] = AttentionWeights[attOffset + (rowBlock*16 + row)*T + col]; } else{ attShared[row*T_padded + col] = 0.0f; }
	}
	__syncthreads();
	for(int col_block = warpId; col_block < (T + 15) / 16; col_block += numWarps){
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			fill_fragment(cFrag, 0.0f);
			if(laneId < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspaceHalf[laneId*16 + i] = __float2half(col_block*16 + laneId < T ? attShared[i*T_padded + col_block*16 + laneId] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(aFrag, workspaceHalf, 16);
			load_matrix_sync(bFrag, dOutShared + d_block*16, D_padded);
			mma_sync(cFrag, aFrag, bFrag, cFrag);
			store_matrix_sync(workspaceFloat, cFrag, 16, wmma::mem_row_major);
			__syncwarp();
#pragma unroll
			for(int i = laneId; i < 16*16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(col_block*16 + row < T && d_block*16 + col < D){ AtomicAddHalf(&dV[batchHeadOffset + (col_block*16 + row)*D + d_block*16 + col], __float2half(workspaceFloat[row*16 + col])); }
			}
		}
	}
	__syncthreads();
	for(int col_block = warpId; col_block < (T + 15) / 16; col_block += numWarps){
		fill_fragment(cFrag, 0.0f);
#pragma unroll
		for(int d = laneId; d < D*16; d += 32){
			int row = d / D;
			int col = d % D;
			if(col_block*16 + row < T && col < D){ vShared[row*D_padded + col] = V[batchHeadOffset + (col_block*16 + row)*D + col]; } else{ vShared[row*D_padded + col] = __float2half(0.0f); }
		}
		__syncwarp();
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
#pragma unroll
			for(int i = laneId; i < 16*16; i += 32){
				int row = i / 16;
				int col = i % 16;
				workspaceHalf[col*16 + row] = vShared[row*D_padded + d_block*16 + col];
			}
			__syncwarp();
			load_matrix_sync(aFrag, dOutShared + d_block*16, D_padded);
			load_matrix_sync(bFrag, workspaceHalf, 16);
			mma_sync(cFrag, aFrag, bFrag, cFrag);
		}
		store_matrix_sync(workspaceFloat, cFrag, 16, wmma::mem_row_major);
		__syncwarp();
#pragma unroll
		for(int i = laneId; i < 16*16; i += 32){
			int row = i / 16;
			int col = i % 16;
			if(col_block*16 + col < T){ dAttShared[row*T_padded + col_block*16 + col] = workspaceFloat[row*16 + col]; }
		}
	}
	__syncthreads();
#pragma unroll
	for(int row = 0; row < 16; row++){
		float sum = 0.0f;
		for(int col = threadIdx.x; col < T; col += blockDim.x){ if(rowBlock*16 + row < T){ sum += dAttShared[row*T_padded + col]*attShared[row*T_padded + col]; } }
#pragma unroll
		for(int offset = 16; offset > 0; offset /= 2){ sum += __shfl_down_sync(0xffffffff, sum, offset); }
		if(laneId == 0){ warpRowsum[row*numWarps + warpId] = sum; }
	}
	__syncthreads();
	if(threadIdx.x < 16){
		float total_sum = 0.0f;
		for(int w = 0; w < numWarps; w++){ total_sum += warpRowsum[threadIdx.x*numWarps + w]; }
		rowsumShared[threadIdx.x] = total_sum;
	}
	__syncthreads();
#pragma unroll
	for(int i = threadIdx.x; i < 16*T; i += blockDim.x){
		int row = i / T;
		int col = i % T;
		if(rowBlock*16 + row < T && col < T){
			float a = attShared[row*T_padded + col];
			float da = dAttShared[row*T_padded + col];
			if(isfinite(a) && isfinite(da) && a > 1e-8f){ dAttShared[row*T_padded + col] = a*(da - rowsumShared[row]); } else{ dAttShared[row*T_padded + col] = 0.0f; }
		}
	}
	__syncthreads();
#pragma unroll
	for(int d = threadIdx.x; d < 16*D; d += blockDim.x){
		int row = d / D;
		int col = d % D;
		if(rowBlock*16 + row < T && col < D){ qShared[row*D_padded + col] = Q[batchHeadOffset + (rowBlock*16 + row)*D + col]; } else{ qShared[row*D_padded + col] = __float2half(0.0f); }
	}
	__syncthreads();
	const float scale = rsqrtf(static_cast<float>(D));
	for(int d_block = warpId; d_block < (D + 15) / 16; d_block += numWarps){
		fill_fragment(cFrag, 0.0f);
		for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
#pragma unroll
			for(int i = laneId; i < 16*16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(col_block*16 + row < T && d_block*16 + col < D){ workspaceHalf[row*16 + col] = K[batchHeadOffset + (col_block*16 + row)*D + d_block*16 + col]; } else{ workspaceHalf[row*16 + col] = __float2half(0.0f); }
			}
			__syncwarp();
			if(laneId < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspaceHalf[256 + i*16 + laneId] = __float2half(col_block*16 + laneId < T ? dAttShared[i*T_padded + col_block*16 + laneId] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(aFrag, workspaceHalf + 256, 16);
			load_matrix_sync(bFrag, workspaceHalf, 16);
			mma_sync(cFrag, aFrag, bFrag, cFrag);
		}
#pragma unroll
		for(int i = 0; i < cFrag.num_elements; i++){ cFrag.x[i] *= scale; }
		store_matrix_sync(workspaceFloat, cFrag, 16, wmma::mem_row_major);
		__syncwarp();
#pragma unroll
		for(int i = laneId; i < 16*16; i += 32){
			int row = i / 16;
			int col = i % 16;
			if(rowBlock*16 + row < T && d_block*16 + col < D){ AtomicAddHalf(&dQ[batchHeadOffset + (rowBlock*16 + row)*D + d_block*16 + col], __float2half(workspaceFloat[row*16 + col])); }
		}
	}
	for(int k_row_block = warpId; k_row_block < (T + 15) / 16; k_row_block += numWarps){
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			fill_fragment(cFrag, 0.0f);
			if(laneId < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspaceHalf[laneId*16 + i] = __float2half(k_row_block*16 + laneId < T ? dAttShared[i*T_padded + k_row_block*16 + laneId] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(aFrag, workspaceHalf, 16);
			load_matrix_sync(bFrag, qShared + d_block*16, D_padded);
			mma_sync(cFrag, aFrag, bFrag, cFrag);
#pragma unroll
			for(int i = 0; i < cFrag.num_elements; i++){ cFrag.x[i] *= scale; }
			store_matrix_sync(workspaceFloat, cFrag, 16, wmma::mem_row_major);
			__syncwarp();
#pragma unroll
			for(int i = laneId; i < 16*16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(k_row_block*16 + row < T && d_block*16 + col < D){ AtomicAddHalf(&dK[batchHeadOffset + (k_row_block*16 + row)*D + d_block*16 + col], __float2half(workspaceFloat[row*16 + col])); }
			}
		}
	}
}
void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* Att, __half* dQ, __half* dK, __half* dV, int B, int T, int D, int H){
	cudaMemset(dQ, 0, B*H*T*D*sizeof(__half));
	cudaMemset(dK, 0, B*H*T*D*sizeof(__half));
	cudaMemset(dV, 0, B*H*T*D*sizeof(__half));
	dim3 block(128);
	dim3 grid((T + 15) / 16, B, H);
	const int dPadded = D + 7 & ~7;
	const int tPadded = T + 3 & ~3;
	constexpr int numWarps = 128 / 32;
	size_t sharedSize = sizeof(__half)*(16*dPadded*4) + sizeof(float)*(16*tPadded*2) + sizeof(float)*16 + sizeof(float)*(16*numWarps) + sizeof(__half)*512 + sizeof(float)*256;
	cudaFuncSetAttribute(WmmaAttentionBackwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	cudaFuncSetAttribute(WmmaAttentionBackwardKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
	WmmaAttentionBackwardKernel<<<grid, block, sharedSize>>>(Q, K, V, dOut, Att, dQ, dK, dV, B, T, D, H);
}
__host__ __device__ inline int RoundUp(int x, int m){ return (x + m - 1) / m*m; }
template <int TILE_M, int TILE_N, int TILE_K, int WARPS_M, int WARPS_N>
__global__ void OptimizedWmmaAttentionBackwardKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ Att, __half* __restrict__ dQ, __half* __restrict__ dK, __half* __restrict__ dV, int B, int T, int D, int H){
	// Constants
	constexpr int wmmaM = 16;
	constexpr int wmmaN = 16;
	constexpr int wmmaK = 16;
	constexpr int warpsPerBlock = WARPS_M*WARPS_N;
	constexpr int threadsPerBlock = warpsPerBlock*32;
	// Thread indices
	const int tid = threadIdx.x;
	const int warpId = tid / 32;
	const int laneId = tid % 32;
	const int warpM = warpId / WARPS_N;
	const int warpN = warpId % WARPS_N;
	// Block indices
	const int blockM = blockIdx.x;
	const int batch = blockIdx.y;
	const int head = blockIdx.z;
	// Early exit
	if(blockM*TILE_M >= T) return;
	// Offsets
	const int batchHeadOffset = (batch*H + head)*T*D;
	const int attOffset = (batch*H + head)*T*T;
	// Shared memory allocation with padding to avoid bank conflicts
	extern __shared__ __align__(16) char smem[];
	// Shared memory pointers with proper alignment
	auto qSmem = reinterpret_cast<__half*>(smem);
	const int dStride = D + 7 & ~7;
	const int tStride = T + 7 & ~7;
	__half* dOutSmem = qSmem + TILE_M*dStride;
	__half* kvSmem = dOutSmem + TILE_M*dStride;
	auto attSmem = reinterpret_cast<float*>(kvSmem + TILE_K*dStride);
	float* dAttSmem = attSmem + TILE_M*tStride;
	float* rowSum = dAttSmem + TILE_M*tStride;
	constexpr int aStorage = wmmaM*wmmaK;
	constexpr int bStorage = wmmaK*wmmaN;
	constexpr int halfStorage = aStorage > bStorage ? aStorage : bStorage;
	constexpr int cStorage = wmmaM*wmmaN;
	auto warpHalfStorage = reinterpret_cast<__half*>(rowSum + TILE_M);
	auto warpFloatStorage = reinterpret_cast<float*>(warpHalfStorage + warpsPerBlock*halfStorage);
	__half* myHalfStorage = warpHalfStorage + warpId*halfStorage;
	float* myFloatStorage = warpFloatStorage + warpId*cStorage;
	// Fragment declarations
	wmma::fragment<wmma::matrix_a, wmmaM, wmmaN, wmmaK, __half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, wmmaM, wmmaN, wmmaK, __half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, wmmaM, wmmaN, wmmaK, float> c_frag;
	const float scale = rsqrtf(static_cast<float>(D));
	// Load Q and dOut for this tile into shared memory
#pragma unroll
	for(int i = tid; i < TILE_M*dStride; i += threadsPerBlock){
		int row = i / dStride;
		int col = i % dStride;
		int global_row = blockM*TILE_M + row;
		if(global_row < T && col < D){
			qSmem[row*dStride + col] = Q[batchHeadOffset + global_row*D + col];
			dOutSmem[row*dStride + col] = dOut[batchHeadOffset + global_row*D + col];
		} else{
			qSmem[row*dStride + col] = __float2half(0.0f);
			dOutSmem[row*dStride + col] = __float2half(0.0f);
		}
	}
	// Load attention weights for this tile
#pragma unroll
	for(int i = tid; i < TILE_M*tStride; i += threadsPerBlock){
		int row = i / tStride;
		int col = i % tStride;
		int globalRow = blockM*TILE_M + row;
		float val = 0.0f;
		if(globalRow < T && col < T){ val = Att[attOffset + globalRow*T + col]; }
		attSmem[row*tStride + col] = val;
	}
	__syncthreads();
	// Step 1: Compute dV = Att^T @ dOut
	// Process in tiles to accumulate results in registers before writing
	for(int kBlock = 0; kBlock < (T + TILE_K - 1) / TILE_K; kBlock++){
		// Each warp computes a WMMA_M x WMMA_N tile
		for(int dBlock = warpN; dBlock < (D + wmmaN - 1) / wmmaN; dBlock += WARPS_N){
			fill_fragment(c_frag, 0.0f);
			for(int mIter = 0; mIter < TILE_M; mIter += wmmaK){
#pragma unroll
				for(int idx = laneId; idx < wmmaM*wmmaK; idx += 32){
					int row = idx / wmmaK;
					int col = idx % wmmaK;
					int attRow = mIter + col;
					int attCol = kBlock*TILE_K + warpM*wmmaM + row;
					float val = 0.0f;
					if(attRow < TILE_M && attCol < T){ val = attSmem[attRow*tStride + attCol]; }
					myHalfStorage[row*wmmaK + col] = __float2half(val);
				}
				__syncwarp();
				load_matrix_sync(a_frag, myHalfStorage, wmmaK);
				load_matrix_sync(b_frag, &dOutSmem[mIter*dStride + dBlock*wmmaN], dStride);
				mma_sync(c_frag, a_frag, b_frag, c_frag);
			}
			// Store to accumulator
			if(kBlock*TILE_K + warpM*wmmaM < T && dBlock*wmmaN < D){
				store_matrix_sync(myFloatStorage, c_frag, wmmaN, wmma::mem_row_major);
				__syncwarp();
				if(laneId == 0){
#pragma unroll
					for(int i = 0; i < wmmaM && kBlock*TILE_K + warpM*wmmaM + i < T; i++){
#pragma unroll
						for(int j = 0; j < wmmaN && dBlock*wmmaN + j < D; j++){
							AtomicAddHalf(&dV[batchHeadOffset + (kBlock*TILE_K + warpM*wmmaM + i)*D + dBlock*wmmaN + j], __float2half(myFloatStorage[i*wmmaN + j]));
						}
					}
				}
				__syncwarp();
			}
		}
	}
	__syncthreads();
	// Step 2: Compute dAtt = dOut @ V^T
#pragma unroll
	for(int i = tid; i < TILE_M*tStride; i += threadsPerBlock){ dAttSmem[i] = 0.0f; }
	__syncthreads();
	for(int kBlock = 0; kBlock < (T + TILE_K - 1) / TILE_K; kBlock++){
		// Load V tile
#pragma unroll
		for(int i = tid; i < TILE_K*dStride; i += threadsPerBlock){
			int row = i / dStride;
			int col = i % dStride;
			int global_row = kBlock*TILE_K + row;
			if(global_row < T && col < D){ kvSmem[row*dStride + col] = V[batchHeadOffset + global_row*D + col]; } else{ kvSmem[row*dStride + col] = __float2half(0.0f); }
		}
		__syncthreads();
		// Compute tiles of dAtt
		int nBlock = kBlock*TILE_K / wmmaN + warpN;
		if(nBlock*wmmaN < T){
			for(int m = warpM; m < TILE_M / wmmaM; m += WARPS_M){
				fill_fragment(c_frag, 0.0f);
				for(int k = 0; k < (D + wmmaK - 1) / wmmaK; k++){
					load_matrix_sync(a_frag, &dOutSmem[m*wmmaM*dStride + k*wmmaK], dStride);
					// Load V transpose
					for(int idx = laneId; idx < wmmaK*wmmaN; idx += 32){
						int row = idx / wmmaN;
						int col = idx % wmmaN;
						int vRow = nBlock*wmmaN + col - kBlock*TILE_K;
						int vCol = k*wmmaK + row;
						__half val = __float2half(0.0f);
						if(vRow >= 0 && vRow < TILE_K && vCol < D){ val = kvSmem[vRow*dStride + vCol]; }
						myHalfStorage[row*wmmaN + col] = val;
					}
					__syncwarp();
					load_matrix_sync(b_frag, myHalfStorage, wmmaN);
					mma_sync(c_frag, a_frag, b_frag, c_frag);
				}
				// Store to shared memory
				store_matrix_sync(myFloatStorage, c_frag, wmmaN, wmma::mem_row_major);
#pragma unroll
				for(int i = 0; i < wmmaM; i++){
#pragma unroll
					for(int j = 0; j < wmmaN && nBlock*wmmaN + j < T; j++){
						int rowIdx = m*wmmaM + i;
						int colIdx = nBlock*wmmaN + j;
						if(rowIdx < TILE_M){
							dAttSmem[rowIdx*tStride + colIdx] = myFloatStorage[i*wmmaN + j];
						}
					}
				}
			}
		}
	}
	__syncthreads();
	// Step 3: Compute row sums for softmax backward
	if(tid < TILE_M){
		float sum = 0.0f;
		if(blockM*TILE_M + tid < T){
#pragma unroll
			for(int j = 0; j < T; j++){
				float datt = dAttSmem[tid*tStride + j];
				float att = attSmem[tid*tStride + j];
				sum += datt*att;
			}
		}
		rowSum[tid] = sum;
	}
	__syncthreads();
	// Step 4: Apply softmax backward
#pragma unroll
	for(int i = tid; i < TILE_M*T; i += threadsPerBlock){
		int row = i / T;
		int col = i % T;
		if(blockM*TILE_M + row < T && col < T){
			float att = attSmem[row*tStride + col];
			float datt = dAttSmem[row*tStride + col];
			float updated = att*(datt - rowSum[row]);
			dAttSmem[row*tStride + col] = updated;
		} else{
			dAttSmem[row*tStride + col] = 0.0f;
		}
	}
	__syncthreads();
	// Step 5: Compute dQ and dK
	// Process K in tiles
	for(int kBlock = 0; kBlock < (T + TILE_K - 1) / TILE_K; kBlock++){
		// Load K tile
#pragma unroll
		for(int i = tid; i < TILE_K*dStride; i += threadsPerBlock){
			int row = i / dStride;
			int col = i % dStride;
			int global_row = kBlock*TILE_K + row;
			if(global_row < T && col < D){ kvSmem[row*dStride + col] = K[batchHeadOffset + global_row*D + col]; } else{ kvSmem[row*dStride + col] = __float2half(0.0f); }
		}
		__syncthreads();
		// Compute dQ tiles
		for(int dBlock = warpN; dBlock < (D + wmmaN - 1) / wmmaN; dBlock += WARPS_N){
			for(int m = warpM; m < TILE_M / wmmaM; m += WARPS_M){
				fill_fragment(c_frag, 0.0f);
				for(int k = 0; k < TILE_K / wmmaK; k++){
					// Load dAtt
					for(int idx = laneId; idx < wmmaM*wmmaK; idx += 32){
						int row = idx / wmmaK;
						int col = idx % wmmaK;
						int t_idx = kBlock*TILE_K + k*wmmaK + col;
						float val = 0.0f;
						if(t_idx < T){ val = dAttSmem[(m*wmmaM + row)*tStride + t_idx]*scale; }
						myHalfStorage[row*wmmaK + col] = __float2half(val);
					}
					__syncwarp();
					load_matrix_sync(a_frag, myHalfStorage, wmmaK);
					load_matrix_sync(b_frag, &kvSmem[k*wmmaK*dStride + dBlock*wmmaN], dStride);
					mma_sync(c_frag, a_frag, b_frag, c_frag);
				}
				// Store to global dQ
				store_matrix_sync(myFloatStorage, c_frag, wmmaN, wmma::mem_row_major);
				__syncwarp();
				if(laneId == 0){
#pragma unroll
					for(int i = 0; i < wmmaM && blockM*TILE_M + m*wmmaM + i < T; i++){
#pragma unroll
						for(int j = 0; j < wmmaN && dBlock*wmmaN + j < D; j++){
							const int qRow = blockM*TILE_M + m*wmmaM + i;
							const int qCol = dBlock*wmmaN + j;
							__half* dst = &dQ[batchHeadOffset + qRow*D + qCol];
							float accum = __half2float(*dst) + myFloatStorage[i*wmmaN + j];
							*dst = __float2half(accum);
						}
					}
				}
				__syncwarp();
			}
		}
		// Compute dK tiles
		for(int dBlock = warpN; dBlock < (D + wmmaN - 1) / wmmaN; dBlock += WARPS_N){
			for(int m = warpM; m < TILE_K / wmmaM; m += WARPS_M){
				const int keyRowBase = kBlock*TILE_K + m*wmmaM;
				if(keyRowBase >= T) continue;
				fill_fragment(c_frag, 0.0f);
				for(int q = 0; q < TILE_M / wmmaM; q++){
					for(int idx = laneId; idx < wmmaM*wmmaK; idx += 32){
						int row = idx / wmmaK;
						int col = idx % wmmaK;
						int queryRow = q*wmmaK + col;
						int globalQuery = blockM*TILE_M + queryRow;
						int globalKey = keyRowBase + row;
						float val = 0.0f;
						if(queryRow < TILE_M && globalQuery < T && globalKey < T){
							val = dAttSmem[queryRow*tStride + globalKey]*scale;
						}
						myHalfStorage[row*wmmaK + col] = __float2half(val);
					}
					__syncwarp();
					load_matrix_sync(a_frag, myHalfStorage, wmmaK);
					load_matrix_sync(b_frag, &qSmem[q*wmmaM*dStride + dBlock*wmmaN], dStride);
					mma_sync(c_frag, a_frag, b_frag, c_frag);
				}
				// Store to global dK
				store_matrix_sync(myFloatStorage, c_frag, wmmaN, wmma::mem_row_major);
				__syncwarp();
				if(laneId == 0){
#pragma unroll
					for(int i = 0; i < wmmaM && kBlock*TILE_K + m*wmmaM + i < T; i++){
#pragma unroll
						for(int j = 0; j < wmmaN && dBlock*wmmaN + j < D; j++){
							AtomicAddHalf(&dK[batchHeadOffset + (kBlock*TILE_K + m*wmmaM + i)*D + dBlock*wmmaN + j], __float2half(myFloatStorage[i*wmmaN + j]));
						}
					}
				}
				__syncwarp();
			}
		}
	}
}
template <int TILE_M, int TILE_N, int TILE_K, int WARPS_M, int WARPS_N> size_t OptimizedBackwardSharedMemBytes(int D, int T){
	const int dStride = D + 7 & ~7;
	const int tStride = T + 7 & ~7;
	constexpr int wmmaM = 16;
	constexpr int wmmaN = 16;
	constexpr int wmmaK = 16;
	constexpr int warpsPerBlock = WARPS_M*WARPS_N;
	constexpr int aStorage = wmmaM*wmmaK;
	constexpr int bStorage = wmmaK*wmmaN;
	constexpr int halfStorage = aStorage > bStorage ? aStorage : bStorage;
	constexpr int cStorage = wmmaM*wmmaN;
	size_t smemSize = 0;
	smemSize += sizeof(__half)*TILE_M*dStride; // Q_smem
	smemSize += sizeof(__half)*TILE_M*dStride; // dOut_smem
	smemSize += sizeof(__half)*TILE_K*dStride; // Shared K/V buffer
	smemSize += sizeof(float)*TILE_M*tStride; // Att_smem
	smemSize += sizeof(float)*TILE_M*tStride; // dAtt_smem
	smemSize += sizeof(float)*TILE_M; // row_sum
	smemSize += sizeof(__half)*warpsPerBlock*halfStorage; // warp half storage
	smemSize += sizeof(float)*warpsPerBlock*cStorage; // warp float storage
	return smemSize;
}

template <int TILE_M, int TILE_N, int TILE_K, int WARPS_M, int WARPS_N>
bool LaunchOptimizedBackwardConfig(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* att, __half* dQ, __half* dK, __half* dV, int B, int T, int D, int H, size_t maxDynamicSmem){
	const size_t smemSize = OptimizedBackwardSharedMemBytes<TILE_M, TILE_N, TILE_K, WARPS_M, WARPS_N>(D, T);
	if(smemSize > maxDynamicSmem){ return false; }
	dim3 grid((T + TILE_M - 1) / TILE_M, B, H);
	dim3 block(WARPS_M*WARPS_N*32);
	checkCUDA(cudaFuncSetAttribute(OptimizedWmmaAttentionBackwardKernel<TILE_M, TILE_N, TILE_K, WARPS_M, WARPS_N>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemSize)));
	checkCUDA(cudaFuncSetAttribute(OptimizedWmmaAttentionBackwardKernel<TILE_M, TILE_N, TILE_K, WARPS_M, WARPS_N>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
	OptimizedWmmaAttentionBackwardKernel<TILE_M, TILE_N, TILE_K, WARPS_M, WARPS_N><<<grid, block, smemSize>>>(Q, K, V, dOut, att, dQ, dK, dV, B, T, D, H);
	checkCUDA(cudaGetLastError());
	return true;
}
// Optimized launch function
void OptimizedWmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* Att, __half* dQ, __half* dK, __half* dV, int B, int T, int D, int H){
	const size_t tensorElems = static_cast<size_t>(B)*H*T*D;
	// Clear output tensors
	checkCUDA(cudaMemsetAsync(dQ, 0, tensorElems*sizeof(__half)));
	checkCUDA(cudaMemsetAsync(dK, 0, tensorElems*sizeof(__half)));
	checkCUDA(cudaMemsetAsync(dV, 0, tensorElems*sizeof(__half)));
	int device = 0;
	checkCUDA(cudaGetDevice(&device));
	int maxDynamicSmem = 0;
	cudaDeviceProp prop{};
	checkCUDA(cudaGetDeviceProperties(&prop, device));
	maxDynamicSmem = prop.sharedMemPerBlock;
	if(prop.sharedMemPerBlockOptin > maxDynamicSmem){ maxDynamicSmem = prop.sharedMemPerBlockOptin; }
	bool launched = false;
	launched = LaunchOptimizedBackwardConfig<64, 64, 32, 4, 2>(Q, K, V, dOut, Att, dQ, dK, dV, B, T, D, H, static_cast<size_t>(maxDynamicSmem));
	if(!launched){
		launched = LaunchOptimizedBackwardConfig<32, 64, 32, 2, 2>(Q, K, V, dOut, Att, dQ, dK, dV, B, T, D, H, static_cast<size_t>(maxDynamicSmem));
	}
	if(!launched){
		launched = LaunchOptimizedBackwardConfig<16, 64, 32, 1, 2>(Q, K, V, dOut, Att, dQ, dK, dV, B, T, D, H, static_cast<size_t>(maxDynamicSmem));
	}
	if(!launched){
		WmmaAttentionBackward(Q, K, V, dOut, Att, dQ, dK, dV, B, T, D, H);
	}
}