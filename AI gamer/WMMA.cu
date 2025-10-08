#include "CuCommon.cuh"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda::wmma;
__global__ void WmmaAttentionKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ Out, float* __restrict__ AttentionWeights, int B, int T, int D, int H){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int row_block = blockIdx.x;
	if(row_block * 16 >= T) return;
	const int warp_id = threadIdx.x / 32;
	const int lane_id = threadIdx.x % 32;
	const int num_warps = blockDim.x / 32;
	const int batch_head_offset = (batch * H + head) * T * D;
	extern __shared__ char shared_mem_bytes[];
	auto Q_shared = (__half*)shared_mem_bytes;
	__half* K_shared = Q_shared + 16 * D;
	__half* K_transposed = K_shared + 16 * D;
	__half* V_tile = K_transposed + 16 * 16;
	auto scores_shared = (float*)(V_tile + 16 * 16);
	float* row_max = scores_shared + 16 * T;
	float* row_sum = row_max + 16;
	auto att_tile = (__half*)(row_sum + 16);
	auto warp_row_max = reinterpret_cast<float*>(att_tile + 16 * 16);
	fragment<matrix_a, 16, 16, 16, __half, row_major> q_frag;
	fragment<matrix_b, 16, 16, 16, __half, col_major> k_frag;
	fragment<matrix_b, 16, 16, 16, __half, row_major> v_frag;
	fragment<accumulator, 16, 16, 16, float> scores_frag;
	fragment<accumulator, 16, 16, 16, float> out_frag;
	if(threadIdx.x < 16){
		row_max[threadIdx.x] = -FLT_MAX;
		row_sum[threadIdx.x] = 0.0f;
	}
#pragma unroll
	for(int d = threadIdx.x; d < D * 16; d += blockDim.x){
		const int row = d / D;
		const int col = d % D;
		if(row_block * 16 + row < T){ Q_shared[row * D + col] = Q[batch_head_offset + (row_block * 16 + row) * D + col]; } else{ Q_shared[row * D + col] = __float2half(0.0f); }
	}
	__syncthreads();
	for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
		fill_fragment(scores_frag, 0.0f);
#pragma unroll
		for(int d = threadIdx.x; d < D * 16; d += blockDim.x){
			const int row = d / D;
			const int col = d % D;
			if(col_block * 16 + row < T){ K_shared[row * D + col] = K[batch_head_offset + (col_block * 16 + row) * D + col]; } else{ K_shared[row * D + col] = __float2half(0.0f); }
		}
		__syncthreads();
#pragma unroll
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			if(warp_id == 0){
#pragma unroll
				for(int i = lane_id; i < 16 * 16; i += 32){
					const int row = i / 16;
					const int col = i % 16;
					if(d_block * 16 + col < D){ K_transposed[col * 16 + row] = K_shared[row * D + d_block * 16 + col]; } else{ K_transposed[col * 16 + row] = __float2half(0.0f); }
				}
			}
			__syncthreads();
			load_matrix_sync(q_frag, Q_shared + d_block * 16, D);
			load_matrix_sync(k_frag, K_transposed, 16);
			mma_sync(scores_frag, q_frag, k_frag, scores_frag);
		}
#pragma unroll
		for(int i = 0; i < scores_frag.num_elements; i++){ scores_frag.x[i] *= rsqrtf(static_cast<float>(D)); }
		if(warp_id == 0){ store_matrix_sync(scores_shared + col_block * 16, scores_frag, T, mem_row_major); }
		__syncthreads();
	}
#pragma unroll
	for(int row = 0; row < 16; row++){
		float m = -FLT_MAX;
		for(int col = threadIdx.x; col < T; col += blockDim.x){ if(row_block * 16 + row < T && col < T){ m = fmaxf(m, scores_shared[row * T + col]); } }
#pragma unroll
		for(int offset = 16; offset > 0; offset /= 2){ m = fmaxf(m, __shfl_down_sync(0xffffffff, m, offset)); }
		if(lane_id == 0){ warp_row_max[row * num_warps + warp_id] = m; }
	}
	__syncthreads();
	if(threadIdx.x < 16){
		float max_val = -FLT_MAX;
		for(int w = 0; w < num_warps; w++){ max_val = fmaxf(max_val, warp_row_max[threadIdx.x * num_warps + w]); }
		row_max[threadIdx.x] = max_val;
	}
	__syncthreads();
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		const int row = i / T;
		const int col = i % T;
		if(row_block * 16 + row < T && col < T){
			const float val = expf(scores_shared[row * T + col] - row_max[row]);
			scores_shared[row * T + col] = val;
			atomicAdd(&row_sum[row], val);
		}
	}
	__syncthreads();
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		const int row = i / T;
		const int col = i % T;
		if(row_block * 16 + row < T && col < T){
			const float normalized = scores_shared[row * T + col] / row_sum[row];
			scores_shared[row * T + col] = normalized;
			AttentionWeights[batch * H * T * T + head * T * T + (row_block * 16 + row) * T + col] = normalized;
		}
	}
	__syncthreads();
	for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
		fill_fragment(out_frag, 0.0f);
#pragma unroll
		for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
			fragment<matrix_a, 16, 16, 16, __half, row_major> att_frag;
			if(warp_id == 0){
#pragma unroll
				for(int i = lane_id; i < 16 * 16; i += 32){
					const int row = i / 16;
					const int col = i % 16;
					if(col_block * 16 + col < T){ att_tile[row * 16 + col] = __float2half(scores_shared[row * T + col_block * 16 + col]); } else{ att_tile[row * 16 + col] = __float2half(0.0f); }
				}
			}
			__syncthreads();
#pragma unroll
			for(int i = threadIdx.x; i < 16 * 16; i += blockDim.x){
				const int row = i / 16;
				const int col = i % 16;
				if(col_block * 16 + row < T && d_block * 16 + col < D){ V_tile[row * 16 + col] = V[batch_head_offset + (col_block * 16 + row) * D + d_block * 16 + col]; } else{ V_tile[row * 16 + col] = __float2half(0.0f); }
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
				if(row_block * 16 + row < T && d_block * 16 + col < D){ Out[batch_head_offset + (row_block * 16 + row) * D + d_block * 16 + col] = __float2half(out_frag.x[i]); }
			}
		}
		__syncthreads();
	}
}
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, float* AttentionWeights, int B, int T, int D, int H){
	dim3 block(128);
	dim3 grid((T + 15) / 16, B, H);
	const int num_warps = block.x / 32;
	size_t shared_size = sizeof(__half) * (16 * D * 2 + 16 * 16 + 16 * 16 + 16 * 16) + sizeof(float) * (16 * T + 16 + 16 + 16 * num_warps);
	cudaFuncSetAttribute(WmmaAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	WmmaAttentionKernel<<<grid, block, shared_size>>>(Q, K, V, Out, AttentionWeights, B, T, D, H);
	const auto e = cudaGetLastError();
	if(e != cudaSuccess) printf("WmmaAttention Forward error: %s\n", cudaGetErrorString(e));
}
__global__ void WmmaAttentionBackwardKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ AttentionWeights, __half* __restrict__ dQ, __half* __restrict__ dK, __half* __restrict__ dV, int B, int T,
											int D, int H){
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int row_block = blockIdx.x;
	if(row_block * 16 >= T) return;
	const int warp_id = threadIdx.x / 32;
	const int lane_id = threadIdx.x % 32;
	const int num_warps = blockDim.x / 32;
	const int batch_head_offset = (batch * H + head) * T * D;
	const int att_offset = (batch * H + head) * T * T;
	extern __shared__ char shared_mem_bytes[];
	auto dOut_shared = reinterpret_cast<__half*>(shared_mem_bytes);
	__half* V_shared = dOut_shared + 16 * ((D + 7) & ~7);
	__half* K_shared = V_shared + 16 * ((D + 7) & ~7);
	__half* Q_shared = K_shared + 16 * ((D + 7) & ~7);
	auto att_shared = reinterpret_cast<float*>(Q_shared + 16 * ((D + 7) & ~7));
	float* dAtt_shared = att_shared + 16 * ((T + 3) & ~3);
	float* rowsum_shared = dAtt_shared + 16 * ((T + 3) & ~3);
	// Add space for warp-level reduction
	float* warp_rowsum = rowsum_shared + 16;
	auto workspace_half = reinterpret_cast<__half*>(warp_rowsum + 16 * num_warps);
	auto workspace_float = reinterpret_cast<float*>(workspace_half + 512);
	fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
	fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
	fragment<accumulator, 16, 16, 16, float> c_frag;
	const int D_padded = (D + 7) & ~7;
	const int T_padded = (T + 3) & ~3;
	if(threadIdx.x < 16){ rowsum_shared[threadIdx.x] = 0.0f; }
#pragma unroll
	for(int d = threadIdx.x; d < 16 * D; d += blockDim.x){
		int row = d / D;
		int col = d % D;
		if(row_block * 16 + row < T && col < D){ dOut_shared[row * D_padded + col] = dOut[batch_head_offset + (row_block * 16 + row) * D + col]; } else{ dOut_shared[row * D_padded + col] = __float2half(0.0f); }
	}
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		int row = i / T;
		int col = i % T;
		if(row_block * 16 + row < T && col < T){ att_shared[row * T_padded + col] = AttentionWeights[att_offset + (row_block * 16 + row) * T + col]; } else{ att_shared[row * T_padded + col] = 0.0f; }
	}
	__syncthreads();
	for(int col_block = warp_id; col_block < (T + 15) / 16; col_block += num_warps){
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			fill_fragment(c_frag, 0.0f);
			if(lane_id < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspace_half[lane_id * 16 + i] = __float2half(col_block * 16 + lane_id < T ? att_shared[i * T_padded + col_block * 16 + lane_id] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(a_frag, workspace_half, 16);
			load_matrix_sync(b_frag, dOut_shared + d_block * 16, D_padded);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
			store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
			__syncwarp();
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(col_block * 16 + row < T && d_block * 16 + col < D){ atomicAdd(&dV[batch_head_offset + (col_block * 16 + row) * D + d_block * 16 + col], __float2half(workspace_float[row * 16 + col])); }
			}
		}
	}
	__syncthreads();
	for(int col_block = warp_id; col_block < (T + 15) / 16; col_block += num_warps){
		fill_fragment(c_frag, 0.0f);
#pragma unroll
		for(int d = lane_id; d < D * 16; d += 32){
			int row = d / D;
			int col = d % D;
			if(col_block * 16 + row < T && col < D){ V_shared[row * D_padded + col] = V[batch_head_offset + (col_block * 16 + row) * D + col]; } else{ V_shared[row * D_padded + col] = __float2half(0.0f); }
		}
		__syncwarp();
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				workspace_half[col * 16 + row] = V_shared[row * D_padded + d_block * 16 + col];
			}
			__syncwarp();
			load_matrix_sync(a_frag, dOut_shared + d_block * 16, D_padded);
			load_matrix_sync(b_frag, workspace_half, 16);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
		}
		store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
		__syncwarp();
#pragma unroll
		for(int i = lane_id; i < 16 * 16; i += 32){
			int row = i / 16;
			int col = i % 16;
			if(col_block * 16 + col < T){ dAtt_shared[row * T_padded + col_block * 16 + col] = workspace_float[row * 16 + col]; }
		}
	}
	__syncthreads();
	// Fixed rowsum reduction
#pragma unroll
	for(int row = 0; row < 16; row++){
		float sum = 0.0f;
		for(int col = threadIdx.x; col < T; col += blockDim.x){ if(row_block * 16 + row < T){ sum += dAtt_shared[row * T_padded + col] * att_shared[row * T_padded + col]; } }
		// Warp-level reduction
#pragma unroll
		for(int offset = 16; offset > 0; offset /= 2){ sum += __shfl_down_sync(0xffffffff, sum, offset); }
		// Each warp writes its partial sum
		if(lane_id == 0){ warp_rowsum[row * num_warps + warp_id] = sum; }
	}
	__syncthreads();
	// Final reduction across warps
	if(threadIdx.x < 16){
		float total_sum = 0.0f;
		for(int w = 0; w < num_warps; w++){ total_sum += warp_rowsum[threadIdx.x * num_warps + w]; }
		rowsum_shared[threadIdx.x] = total_sum;
	}
	__syncthreads();
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		int row = i / T;
		int col = i % T;
		if(row_block * 16 + row < T && col < T){
			float a = att_shared[row * T_padded + col];
			float da = dAtt_shared[row * T_padded + col];
			if(isfinite(a) && isfinite(da) && a > 1e-8f){ dAtt_shared[row * T_padded + col] = a * (da - rowsum_shared[row]); } else{ dAtt_shared[row * T_padded + col] = 0.0f; }
		}
	}
	__syncthreads();
#pragma unroll
	for(int d = threadIdx.x; d < 16 * D; d += blockDim.x){
		int row = d / D;
		int col = d % D;
		if(row_block * 16 + row < T && col < D){ Q_shared[row * D_padded + col] = Q[batch_head_offset + (row_block * 16 + row) * D + col]; } else{ Q_shared[row * D_padded + col] = __float2half(0.0f); }
	}
	__syncthreads();
	const float scale = rsqrtf(static_cast<float>(D));
	for(int d_block = warp_id; d_block < (D + 15) / 16; d_block += num_warps){
		fill_fragment(c_frag, 0.0f);
		for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(col_block * 16 + row < T && d_block * 16 + col < D){ workspace_half[row * 16 + col] = K[batch_head_offset + (col_block * 16 + row) * D + d_block * 16 + col]; } else{ workspace_half[row * 16 + col] = __float2half(0.0f); }
			}
			__syncwarp();
			if(lane_id < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspace_half[256 + i * 16 + lane_id] = __float2half(col_block * 16 + lane_id < T ? dAtt_shared[i * T_padded + col_block * 16 + lane_id] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(a_frag, workspace_half + 256, 16);
			load_matrix_sync(b_frag, workspace_half, 16);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
		}
#pragma unroll
		for(int i = 0; i < c_frag.num_elements; i++){ c_frag.x[i] *= scale; }
		store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
		__syncwarp();
#pragma unroll
		for(int i = lane_id; i < 16 * 16; i += 32){
			int row = i / 16;
			int col = i % 16;
			if(row_block * 16 + row < T && d_block * 16 + col < D){ atomicAdd(&dQ[batch_head_offset + (row_block * 16 + row) * D + d_block * 16 + col], __float2half(workspace_float[row * 16 + col])); }
		}
	}
	for(int k_row_block = warp_id; k_row_block < (T + 15) / 16; k_row_block += num_warps){
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			fill_fragment(c_frag, 0.0f);
			if(lane_id < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspace_half[lane_id * 16 + i] = __float2half(k_row_block * 16 + lane_id < T ? dAtt_shared[i * T_padded + k_row_block * 16 + lane_id] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(a_frag, workspace_half, 16);
			load_matrix_sync(b_frag, Q_shared + d_block * 16, D_padded);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
#pragma unroll
			for(int i = 0; i < c_frag.num_elements; i++){ c_frag.x[i] *= scale; }
			store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
			__syncwarp();
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(k_row_block * 16 + row < T && d_block * 16 + col < D){ atomicAdd(&dK[batch_head_offset + (k_row_block * 16 + row) * D + d_block * 16 + col], __float2half(workspace_float[row * 16 + col])); }
			}
		}
	}
}
void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* AttentionWeights, __half* dQ, __half* dK, __half* dV, int B, int T, int D, int H){
	cudaMemset(dQ, 0, B * H * T * D * sizeof(__half));
	cudaMemset(dK, 0, B * H * T * D * sizeof(__half));
	cudaMemset(dV, 0, B * H * T * D * sizeof(__half));
	dim3 block(128);
	dim3 grid((T + 15) / 16, B, H);
	const int D_padded = (D + 7) & ~7;
	const int T_padded = (T + 3) & ~3;
	const int num_warps = 128 / 32; // 4 warps
	size_t shared_size = sizeof(__half) * (16 * D_padded * 4) + sizeof(float) * (16 * T_padded * 2) + sizeof(float) * 16 + sizeof(float) * (16 * num_warps) + sizeof(__half) * 512 + sizeof(float) * 256;
	cudaFuncSetAttribute(WmmaAttentionBackwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	cudaFuncSetAttribute(WmmaAttentionBackwardKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
	WmmaAttentionBackwardKernel<<<grid, block, shared_size>>>(Q, K, V, dOut, AttentionWeights, dQ, dK, dV, B, T, D, H);
}
using namespace nvcuda;
// Optimized kernel using larger tiles and better memory patterns
template <int TILE_M, int TILE_N, int TILE_K, int WARPS_M, int WARPS_N> __global__ void __launch_bounds__(256)
OptimizedWmmaAttentionBackwardKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ AttentionWeights, __half* __restrict__ dQ, __half* __restrict__ dK, __half* __restrict__ dV, int B, int T, int D,
									int H){
	// Constants
	constexpr int WMMA_M = 16;
	constexpr int WMMA_N = 16;
	constexpr int WMMA_K = 16;
	constexpr int WARPS_PER_BLOCK = WARPS_M * WARPS_N;
	constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
	// Thread indices
	const int tid = threadIdx.x;
	const int warp_id = tid / 32;
	const int lane_id = tid % 32;
	const int warp_m = warp_id / WARPS_N;
	const int warp_n = warp_id % WARPS_N;
	// Block indices
	const int block_m = blockIdx.x;
	const int batch = blockIdx.y;
	const int head = blockIdx.z;
	// Early exit
	if(block_m * TILE_M >= T) return;
	// Offsets
	const int batch_head_offset = (batch * H + head) * T * D;
	const int att_offset = (batch * H + head) * T * T;
	// Shared memory allocation with padding to avoid bank conflicts
	extern __shared__ __align__(16) char smem[];
	// Shared memory pointers with proper alignment
	auto Q_smem = reinterpret_cast<__half*>(smem);
	__half* K_smem = Q_smem + TILE_M * (D + 8);
	__half* V_smem = K_smem + TILE_K * (D + 8);
	__half* dOut_smem = V_smem + TILE_K * (D + 8);
	float* Att_smem = reinterpret_cast<float*>(dOut_smem + TILE_M * (D + 8));
	float* dAtt_smem = Att_smem + TILE_M * (T + 8);
	float* row_sum = dAtt_smem + TILE_M * (T + 8);
	// Fragment declarations
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
	const float scale = rsqrtf(static_cast<float>(D));
	// Load Q and dOut for this tile into shared memory
#pragma unroll
	for(int i = tid; i < TILE_M * D; i += THREADS_PER_BLOCK){
		int row = i / D;
		int col = i % D;
		int global_row = block_m * TILE_M + row;
		if(global_row < T && col < D){
			Q_smem[row * (D + 8) + col] = Q[batch_head_offset + global_row * D + col];
			dOut_smem[row * (D + 8) + col] = dOut[batch_head_offset + global_row * D + col];
		} else{
			Q_smem[row * (D + 8) + col] = __float2half(0.0f);
			dOut_smem[row * (D + 8) + col] = __float2half(0.0f);
		}
	}
	// Load attention weights for this tile
#pragma unroll
	for(int i = tid; i < TILE_M * T; i += THREADS_PER_BLOCK){
		int row = i / T;
		int col = i % T;
		int global_row = block_m * TILE_M + row;
		if(global_row < T && col < T){ Att_smem[row * (T + 8) + col] = AttentionWeights[att_offset + global_row * T + col]; } else{ Att_smem[row * (T + 8) + col] = 0.0f; }
	}
	__syncthreads();
	// Step 1: Compute dV = Att^T @ dOut
	// Process in tiles to accumulate results in registers before writing
	float dV_accum[4][4] = {0.0f}; // Local accumulator for each thread
	for(int k_block = 0; k_block < (T + TILE_K - 1) / TILE_K; k_block++){
		// Load K tile for current k_block
#pragma unroll
		for(int i = tid; i < TILE_K * D; i += THREADS_PER_BLOCK){
			int row = i / D;
			int col = i % D;
			int global_row = k_block * TILE_K + row;
			if(global_row < T && col < D){ K_smem[row * (D + 8) + col] = V[batch_head_offset + global_row * D + col]; } else{ K_smem[row * (D + 8) + col] = __float2half(0.0f); }
		}
		__syncthreads();
		// Each warp computes a WMMA_M x WMMA_N tile
		for(int d_block = warp_n; d_block < (D + WMMA_N - 1) / WMMA_N; d_block += WARPS_N){
			wmma::fill_fragment(c_frag, 0.0f);
			// Compute Att^T @ dOut
			for(int m = 0; m < TILE_M / WMMA_M; m++){
				// Load Att transpose (col_major)
				float att_tile[WMMA_M * WMMA_K];
#pragma unroll
				for(int i = 0; i < WMMA_M; i++){
#pragma unroll
					for(int j = 0; j < WMMA_K; j++){
						int att_row = m * WMMA_M + i;
						int att_col = k_block * TILE_K + warp_m * WMMA_K + j;
						att_tile[j * WMMA_M + i] = (att_col < T) ? Att_smem[att_row * (T + 8) + att_col] : 0.0f;
					}
				}
				// Convert to half
				__half att_tile_half[WMMA_M * WMMA_K];
#pragma unroll
				for(int i = 0; i < WMMA_M * WMMA_K; i++){ att_tile_half[i] = __float2half(att_tile[i]); }
				wmma::load_matrix_sync(a_frag, att_tile_half, WMMA_M);
				wmma::load_matrix_sync(b_frag, &dOut_smem[m * WMMA_M * (D + 8) + d_block * WMMA_N], D + 8);
				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
			}
			// Store to accumulator
			if(k_block * TILE_K + warp_m * WMMA_M < T && d_block * WMMA_N < D){
				wmma::store_matrix_sync(&dV_accum[0][0], c_frag, 4, wmma::mem_row_major);
				// Write to global memory with coalesced access
#pragma unroll
				for(int i = 0; i < WMMA_M && k_block * TILE_K + warp_m * WMMA_M + i < T; i++){
#pragma unroll
					for(int j = 0; j < WMMA_N && d_block * WMMA_N + j < D; j++){ atomicAdd(&dV[batch_head_offset + (k_block * TILE_K + warp_m * WMMA_M + i) * D + d_block * WMMA_N + j], __float2half(dV_accum[i][j])); }
				}
			}
		}
	}
	__syncthreads();
	// Step 2: Compute dAtt = dOut @ V^T
#pragma unroll
	for(int i = tid; i < TILE_M * T; i += THREADS_PER_BLOCK){ dAtt_smem[i] = 0.0f; }
	__syncthreads();
	for(int k_block = 0; k_block < (T + TILE_K - 1) / TILE_K; k_block++){
		// Load V tile
#pragma unroll
		for(int i = tid; i < TILE_K * D; i += THREADS_PER_BLOCK){
			int row = i / D;
			int col = i % D;
			int global_row = k_block * TILE_K + row;
			if(global_row < T && col < D){ V_smem[row * (D + 8) + col] = V[batch_head_offset + global_row * D + col]; } else{ V_smem[row * (D + 8) + col] = __float2half(0.0f); }
		}
		__syncthreads();
		// Compute tiles of dAtt
		int n_block = k_block * TILE_K / WMMA_N + warp_n;
		if(n_block * WMMA_N < T){
			for(int m = warp_m; m < TILE_M / WMMA_M; m += WARPS_M){
				wmma::fill_fragment(c_frag, 0.0f);
				for(int k = 0; k < (D + WMMA_K - 1) / WMMA_K; k++){
					wmma::load_matrix_sync(a_frag, &dOut_smem[m * WMMA_M * (D + 8) + k * WMMA_K], D + 8);
					// Load V transpose
					__half v_tile[WMMA_K * WMMA_N];
#pragma unroll
					for(int i = 0; i < WMMA_K && k * WMMA_K + i < D; i++){
#pragma unroll
						for(int j = 0; j < WMMA_N && n_block * WMMA_N + j < T; j++){ v_tile[i * WMMA_N + j] = V_smem[j * (D + 8) + k * WMMA_K + i]; }
					}
					wmma::load_matrix_sync(b_frag, v_tile, WMMA_N);
					wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
				}
				// Store to shared memory
				float temp[WMMA_M * WMMA_N];
				wmma::store_matrix_sync(temp, c_frag, WMMA_N, wmma::mem_row_major);
#pragma unroll
				for(int i = 0; i < WMMA_M; i++){
#pragma unroll
					for(int j = 0; j < WMMA_N && n_block * WMMA_N + j < T; j++){ atomicAdd(&dAtt_smem[(m * WMMA_M + i) * (T + 8) + n_block * WMMA_N + j], temp[i * WMMA_N + j]); }
				}
			}
		}
	}
	__syncthreads();
	// Step 3: Compute row sums for softmax backward
	if(tid < TILE_M){
		float sum = 0.0f;
#pragma unroll
		for(int j = 0; j < T; j++){ sum += dAtt_smem[tid * (T + 8) + j] * Att_smem[tid * (T + 8) + j]; }
		row_sum[tid] = sum;
	}
	__syncthreads();
	// Step 4: Apply softmax backward
#pragma unroll
	for(int i = tid; i < TILE_M * T; i += THREADS_PER_BLOCK){
		int row = i / T;
		int col = i % T;
		if(block_m * TILE_M + row < T && col < T){
			float att = Att_smem[row * (T + 8) + col];
			float datt = dAtt_smem[row * (T + 8) + col];
			dAtt_smem[row * (T + 8) + col] = att * (datt - row_sum[row]);
		}
	}
	__syncthreads();
	// Step 5: Compute dQ and dK
	// Process K in tiles
	for(int k_block = 0; k_block < (T + TILE_K - 1) / TILE_K; k_block++){
		// Load K tile
#pragma unroll
		for(int i = tid; i < TILE_K * D; i += THREADS_PER_BLOCK){
			int row = i / D;
			int col = i % D;
			int global_row = k_block * TILE_K + row;
			if(global_row < T && col < D){ K_smem[row * (D + 8) + col] = K[batch_head_offset + global_row * D + col]; } else{ K_smem[row * (D + 8) + col] = __float2half(0.0f); }
		}
		__syncthreads();
		// Compute dQ tiles
		for(int d_block = warp_n; d_block < (D + WMMA_N - 1) / WMMA_N; d_block += WARPS_N){
			for(int m = warp_m; m < TILE_M / WMMA_M; m += WARPS_M){
				wmma::fill_fragment(c_frag, 0.0f);
				for(int k = 0; k < TILE_K / WMMA_K; k++){
					// Load dAtt
					float datt_tile[WMMA_M * WMMA_K];
#pragma unroll
					for(int i = 0; i < WMMA_M; i++){
#pragma unroll
						for(int j = 0; j < WMMA_K && k_block * TILE_K + k * WMMA_K + j < T; j++){ datt_tile[i * WMMA_K + j] = dAtt_smem[(m * WMMA_M + i) * (T + 8) + k_block * TILE_K + k * WMMA_K + j]/scale; }
					}
					__half datt_tile_half[WMMA_M * WMMA_K];
#pragma unroll
					for(int i = 0; i < WMMA_M * WMMA_K; i++){ datt_tile_half[i] = __float2half(datt_tile[i]); }
					wmma::load_matrix_sync(a_frag, datt_tile_half, WMMA_K);
					wmma::load_matrix_sync(b_frag, &K_smem[k * WMMA_K * (D + 8) + d_block * WMMA_N], D + 8);
					wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
				}
				// Store to global dQ
				float temp[WMMA_M * WMMA_N];
				wmma::store_matrix_sync(temp, c_frag, WMMA_N, wmma::mem_row_major);
#pragma unroll
				for(int i = 0; i < WMMA_M && block_m * TILE_M + m * WMMA_M + i < T; i++){
#pragma unroll
					for(int j = 0; j < WMMA_N && d_block * WMMA_N + j < D; j++){ atomicAdd(&dQ[batch_head_offset + (block_m * TILE_M + m * WMMA_M + i) * D + d_block * WMMA_N + j], __float2half(temp[i * WMMA_N + j])); }
				}
			}
		}
		// Compute dK tiles  
		for(int d_block = warp_n; d_block < (D + WMMA_N - 1) / WMMA_N; d_block += WARPS_N){
			for(int m = warp_m; m < TILE_K / WMMA_M; m += WARPS_M){
				if(k_block * TILE_K + m * WMMA_M >= T) continue;
				wmma::fill_fragment(c_frag, 0.0f);
				// Load dAtt transpose and scale
				float datt_tile[WMMA_M * WMMA_K];
#pragma unroll
				for(int i = 0; i < WMMA_M && block_m * TILE_M + i < T; i++){
#pragma unroll
					for(int j = 0; j < WMMA_K && k_block * TILE_K + m * WMMA_M + j < T; j++){ datt_tile[j * WMMA_M + i] = dAtt_smem[i * (T + 8) + k_block * TILE_K + m * WMMA_M + j]/scale; }
				}
				__half datt_tile_half[WMMA_M * WMMA_K];
#pragma unroll
				for(int i = 0; i < WMMA_M * WMMA_K; i++){ datt_tile_half[i] = __float2half(datt_tile[i]); }
				wmma::load_matrix_sync(a_frag, datt_tile_half, WMMA_M);
				wmma::load_matrix_sync(b_frag, &Q_smem[0 * (D + 8) + d_block * WMMA_N], D + 8);
				for(int q = 1; q < TILE_M / WMMA_M; q++){
					wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
					wmma::load_matrix_sync(b_frag, &Q_smem[q * WMMA_M * (D + 8) + d_block * WMMA_N], D + 8);
				}
				wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
				// Store to global dK
				float temp[WMMA_M * WMMA_N];
				wmma::store_matrix_sync(temp, c_frag, WMMA_N, wmma::mem_row_major);
#pragma unroll
				for(int i = 0; i < WMMA_M && k_block * TILE_K + m * WMMA_M + i < T; i++){
#pragma unroll
					for(int j = 0; j < WMMA_N && d_block * WMMA_N + j < D; j++){ atomicAdd(&dK[batch_head_offset + (k_block * TILE_K + m * WMMA_M + i) * D + d_block * WMMA_N + j], __float2half(temp[i * WMMA_N + j])); }
				}
			}
		}
	}
}
// Optimized launch function
void OptimizedWmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* AttentionWeights, __half* dQ, __half* dK, __half* dV, int B, int T, int D, int H){
	// Clear output tensors
	cudaMemsetAsync(dQ, 0, B * H * T * D * sizeof(__half));
	cudaMemsetAsync(dK, 0, B * H * T * D * sizeof(__half));
	cudaMemsetAsync(dV, 0, B * H * T * D * sizeof(__half));
	// Use larger tiles for better performance
	constexpr int TILE_M = 64;
	constexpr int TILE_N = 64;
	constexpr int TILE_K = 32;
	constexpr int WARPS_M = 4;
	constexpr int WARPS_N = 2;
	dim3 grid((T + TILE_M - 1) / TILE_M, B, H);
	dim3 block(WARPS_M * WARPS_N * 32);
	// Calculate shared memory size
	size_t smem_size = 0;
	smem_size += sizeof(__half) * TILE_M * (D + 8); // Q_smem
	smem_size += sizeof(__half) * TILE_K * (D + 8); // K_smem  
	smem_size += sizeof(__half) * TILE_K * (D + 8); // V_smem
	smem_size += sizeof(__half) * TILE_M * (D + 8); // dOut_smem
	smem_size += sizeof(float) * TILE_M * (T + 8); // Att_smem
	smem_size += sizeof(float) * TILE_M * (T + 8); // dAtt_smem
	smem_size += sizeof(float) * TILE_M; // row_sum
	// Configure kernel attributes
	cudaFuncSetAttribute(OptimizedWmmaAttentionBackwardKernel<TILE_M, TILE_N, TILE_K, WARPS_M, WARPS_N>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
	// For better occupancy on Volta
	cudaFuncSetAttribute(OptimizedWmmaAttentionBackwardKernel<TILE_M, TILE_N, TILE_K, WARPS_M, WARPS_N>, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
	// Launch kernel
	OptimizedWmmaAttentionBackwardKernel<TILE_M, TILE_N, TILE_K, WARPS_M, WARPS_N><<<grid, block, smem_size>>>(Q, K, V, dOut, AttentionWeights, dQ, dK, dV, B, T, D, H);
}