#include "CuCommon.cuh"
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda::wmma;
// Optimized WMMA Attention Kernel for Volta
__global__ void WmmaAttentionKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, __half* __restrict__ Out, float* __restrict__ AttentionWeights, int B, int T, int D, int H){
	// Block indices
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int row_block = blockIdx.x;
	// Early exit
	if(row_block * 16 >= T) return;
	// Warp and lane IDs
	const int warp_id = threadIdx.x / 32;
	const int lane_id = threadIdx.x % 32;
	const int num_warps = blockDim.x / 32;
	// Calculate base offsets
	const int batch_head_offset = (batch * H + head) * T * D;
	// Shared memory layout
	extern __shared__ char shared_mem_bytes[];
	__half* Q_shared = (__half*)shared_mem_bytes;
	__half* K_shared = Q_shared + 16 * D;
	__half* V_shared = K_shared + 16 * D;
	float* scores_shared = (float*)(V_shared + 16 * D);
	float* row_max = scores_shared + 16 * T;
	float* row_sum = row_max + 16;
	// WMMA fragments
	fragment<matrix_a, 16, 16, 16, __half, row_major> q_frag;
	fragment<matrix_b, 16, 16, 16, __half, col_major> k_frag;
	fragment<matrix_b, 16, 16, 16, __half, row_major> v_frag;
	fragment<accumulator, 16, 16, 16, float> scores_frag;
	fragment<accumulator, 16, 16, 16, float> out_frag;
	// Initialize row max and sum
	if(threadIdx.x < 16){
		row_max[threadIdx.x] = -FLT_MAX;
		row_sum[threadIdx.x] = 0.0f;
	}
	// Load Q tile for this row block
#pragma unroll
	for(int d = threadIdx.x; d < D * 16; d += blockDim.x){
		const int row = d / D;
		const int col = d % D;
		if(row_block * 16 + row < T){ Q_shared[row * D + col] = Q[batch_head_offset + (row_block * 16 + row) * D + col]; } else{ Q_shared[row * D + col] = __float2half(0.0f); }
	}
	__syncthreads();
	// Step 1: Compute Q*K^T scores
	// Distribute column blocks across warps
	for(int col_block_base = 0; col_block_base < (T + 15) / 16; col_block_base += num_warps){
		const int col_block = col_block_base + warp_id;
		if(col_block < (T + 15) / 16){
			// Initialize scores accumulator
			fill_fragment(scores_frag, 0.0f);
			// Load K tile into shared memory (transposed for col_major)
#pragma unroll
			for(int d = lane_id; d < D * 16; d += 32){
				const int row = d / D;
				const int col = d % D;
				if(col_block * 16 + row < T){ K_shared[col * 16 + row] = K[batch_head_offset + (col_block * 16 + row) * D + col]; } else{ K_shared[col * 16 + row] = __float2half(0.0f); }
			}
			__syncwarp();
			// Compute Q*K^T using WMMA
#pragma unroll
			for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
				load_matrix_sync(q_frag, Q_shared + d_block * 16, D);
				load_matrix_sync(k_frag, K_shared + d_block * 16 * 16, 16);
				mma_sync(scores_frag, q_frag, k_frag, scores_frag);
			}
			// Scale scores and store
#pragma unroll
			for(int i = 0; i < scores_frag.num_elements; i++){ scores_frag.x[i] *= rsqrtf(static_cast<float>(D)); }
			// Store scores to shared memory
			store_matrix_sync(scores_shared + warp_id * 16 * 16, scores_frag, 16, mem_row_major);
		}
	}
	__syncthreads();
	// Find row-wise max
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		const int row = i / T;
		const int col = i % T;
		if(row_block * 16 + row < T && col < T){
			const float score = scores_shared[row * T + col];
			atomicMax(reinterpret_cast<int*>(&row_max[row]), __float_as_int(score));
		}
	}
	__syncthreads();
	// Convert atomicMax result back to float
	if(threadIdx.x < 16){ row_max[threadIdx.x] = __int_as_float(static_cast<int>(row_max[threadIdx.x])); }
	__syncthreads();
	// Compute exp and sum
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
	// Normalize
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		const int row = i / T;
		const int col = i % T;
		if(row_block * 16 + row < T && col < T){
			const float normalized = scores_shared[row * T + col] / row_sum[row];
			scores_shared[row * T + col] = normalized;
			// Store attention weights
			AttentionWeights[batch * H * T * T + head * T * T + (row_block * 16 + row) * T + col] = normalized;
		}
	}
	__syncthreads();
	// Step 3: Compute attention output
	// Distribute D blocks across warps
	for(int d_block_base = 0; d_block_base < (D + 15) / 16; d_block_base += num_warps){
		const int d_block = d_block_base + warp_id;
		if(d_block < (D + 15) / 16){
			fill_fragment(out_frag, 0.0f);
#pragma unroll
			for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
				// Load attention scores as matrix A
				fragment<matrix_a, 16, 16, 16, __half, row_major> att_frag;
				// Convert float scores to half
				const float* scores_ptr = scores_shared + col_block * 16;
#pragma unroll
				for(int i = 0; i < att_frag.num_elements; i++){
					const int idx = (i / 16) * T + (i % 16);
					att_frag.x[i] = __float2half(scores_ptr[idx]);
				}
				// Load V tile
#pragma unroll
				for(int d = lane_id; d < 16 * 16; d += 32){
					const int row = d / 16;
					const int col = d % 16;
					if(col_block * 16 + row < T && d_block * 16 + col < D){ V_shared[row * 16 + col] = V[batch_head_offset + (col_block * 16 + row) * D + d_block * 16 + col]; } else{ V_shared[row * 16 + col] = __float2half(0.0f); }
				}
				__syncwarp();
				load_matrix_sync(v_frag, V_shared, 16);
				mma_sync(out_frag, att_frag, v_frag, out_frag);
			}
			// Store output
#pragma unroll
			for(int i = 0; i < out_frag.num_elements; i++){
				const int row = i / 16;
				const int col = i % 16;
				if(row_block * 16 + row < T && d_block * 16 + col < D){ Out[batch_head_offset + (row_block * 16 + row) * D + d_block * 16 + col] = __float2half(out_frag.x[i]); }
			}
		}
	}
}
void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, float* AttentionWeights, int B, int T, int D, int H){
	// Use 128 threads (4 warps) for better parallelism
	dim3 block(128);
	dim3 grid((T + 15) / 16, B, H);
	// Calculate shared memory size
	size_t shared_size = sizeof(__half) * (16 * D * 3) + // Q, K, V tiles
		sizeof(float) * (16 * T) + // scores
		sizeof(float) * 32; // row_max + row_sum
	// Check shared memory limit for Volta (96KB max)
	cudaFuncSetAttribute(WmmaAttentionKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	WmmaAttentionKernel<<<grid, block, shared_size>>>(Q, K, V, Out, AttentionWeights, B, T, D, H);
}
// Optimized WMMA Attention Backward Kernel for Volta
__global__ void WmmaAttentionBackwardKernel(const __half* __restrict__ Q, const __half* __restrict__ K, const __half* __restrict__ V, const __half* __restrict__ dOut, const float* __restrict__ AttentionWeights, __half* __restrict__ dQ, __half* __restrict__ dK, __half* __restrict__ dV, int B, int T, int D, int H){
	// Block indices
	const int head = blockIdx.z;
	const int batch = blockIdx.y;
	const int row_block = blockIdx.x;
	// Early exit
	if(row_block * 16 >= T) return;
	// Warp and lane IDs
	const int warp_id = threadIdx.x / 32;
	const int lane_id = threadIdx.x % 32;
	const int num_warps = blockDim.x / 32;
	// Calculate base offsets
	const int batch_head_offset = (batch * H + head) * T * D;
	const int att_offset = (batch * H + head) * T * T;
	// Shared memory layout optimized for bank conflict avoidance
	extern __shared__ char shared_mem_bytes[];
	__half* dOut_shared = reinterpret_cast<__half*>(shared_mem_bytes);
	__half* V_shared = dOut_shared + 16 * ((D + 7) & ~7); // Pad to avoid bank conflicts
	__half* K_shared = V_shared + 16 * ((D + 7) & ~7);
	__half* Q_shared = K_shared + 16 * ((D + 7) & ~7);
	float* att_shared = reinterpret_cast<float*>(Q_shared + 16 * ((D + 7) & ~7));
	float* dAtt_shared = att_shared + 16 * ((T + 3) & ~3); // Pad for alignment
	float* rowsum_shared = dAtt_shared + 16 * ((T + 3) & ~3);
	__half* workspace_half = reinterpret_cast<__half*>(rowsum_shared + 16);
	float* workspace_float = reinterpret_cast<float*>(workspace_half + 512);
	// WMMA fragments
	fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
	fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
	fragment<accumulator, 16, 16, 16, float> c_frag;
	const int D_padded = (D + 7) & ~7;
	const int T_padded = (T + 3) & ~3;
	// Initialize rowsum
	if(threadIdx.x < 16){ rowsum_shared[threadIdx.x] = 0.0f; }
	// Load dOut tile with coalesced access
#pragma unroll
	for(int d = threadIdx.x; d < 16 * D; d += blockDim.x){
		int row = d / D;
		int col = d % D;
		if(row_block * 16 + row < T && col < D){ dOut_shared[row * D_padded + col] = dOut[batch_head_offset + (row_block * 16 + row) * D + col]; } else{ dOut_shared[row * D_padded + col] = __float2half(0.0f); }
	}
	// Load attention weights with coalesced access
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		int row = i / T;
		int col = i % T;
		if(row_block * 16 + row < T && col < T){ att_shared[row * T_padded + col] = AttentionWeights[att_offset + (row_block * 16 + row) * T + col]; } else{ att_shared[row * T_padded + col] = 0.0f; }
	}
	__syncthreads();
	// Step 1: Compute dV = A^T * dOut
	// Use shared memory accumulation to reduce atomics
	for(int col_block = warp_id; col_block < (T + 15) / 16; col_block += num_warps){
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			fill_fragment(c_frag, 0.0f);
			// Load attention weights for WMMA (transposed view)
			if(lane_id < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspace_half[lane_id * 16 + i] = __float2half(col_block * 16 + lane_id < T ? att_shared[i * T_padded + col_block * 16 + lane_id] : 0.0f); }
			}
			__syncwarp();
			// Load fragments and compute
			load_matrix_sync(a_frag, workspace_half, 16);
			load_matrix_sync(b_frag, dOut_shared + d_block * 16, D_padded);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
			// Store to float workspace first
			store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
			__syncwarp();
			// Coalesced write to global memory - convert float to half
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(col_block * 16 + row < T && d_block * 16 + col < D){ atomicAdd(&dV[batch_head_offset + (col_block * 16 + row) * D + d_block * 16 + col], __float2half(workspace_float[row * 16 + col])); }
			}
		}
	}
	__syncthreads();
	// Step 2: Compute dA = dOut * V^T
	for(int col_block = warp_id; col_block < (T + 15) / 16; col_block += num_warps){
		fill_fragment(c_frag, 0.0f);
		// Load V tile with coalesced access
#pragma unroll
		for(int d = lane_id; d < D * 16; d += 32){
			int row = d / D;
			int col = d % D;
			if(col_block * 16 + row < T && col < D){ V_shared[row * D_padded + col] = V[batch_head_offset + (col_block * 16 + row) * D + col]; } else{ V_shared[row * D_padded + col] = __float2half(0.0f); }
		}
		__syncwarp();
		// Compute dOut * V^T using multiple WMMA operations
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			// Prepare V transpose in workspace
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
		// Store result to float workspace
		store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
		__syncwarp();
		// Copy to dAtt_shared (already float)
#pragma unroll
		for(int i = lane_id; i < 16 * 16; i += 32){
			int row = i / 16;
			int col = i % 16;
			if(col_block * 16 + col < T){ dAtt_shared[row * T_padded + col_block * 16 + col] = workspace_float[row * 16 + col]; }
		}
	}
	__syncthreads();
	// Step 3: Softmax backward - compute row sums
#pragma unroll
	for(int row = 0; row < 16; row++){
		float sum = 0.0f;
		for(int col = threadIdx.x; col < T; col += blockDim.x){ if(row_block * 16 + row < T){ sum += dAtt_shared[row * T_padded + col] * att_shared[row * T_padded + col]; } }
		// Warp reduction
#pragma unroll
		for(int offset = 16; offset > 0; offset /= 2){ sum += __shfl_down_sync(0xffffffff, sum, offset); }
		if(lane_id == 0 && threadIdx.x / 32 == 0){ rowsum_shared[row] = sum; }
	}
	__syncthreads();
	// Apply softmax backward
#pragma unroll
	for(int i = threadIdx.x; i < 16 * T; i += blockDim.x){
		int row = i / T;
		int col = i % T;
		if(row_block * 16 + row < T && col < T){
			float a = att_shared[row * T_padded + col];
			float da = dAtt_shared[row * T_padded + col];
			dAtt_shared[row * T_padded + col] = a * (da - rowsum_shared[row]);
		}
	}
	__syncthreads();
	// Load Q tile
#pragma unroll
	for(int d = threadIdx.x; d < 16 * D; d += blockDim.x){
		int row = d / D;
		int col = d % D;
		if(row_block * 16 + row < T && col < D){ Q_shared[row * D_padded + col] = Q[batch_head_offset + (row_block * 16 + row) * D + col]; } else{ Q_shared[row * D_padded + col] = __float2half(0.0f); }
	}
	__syncthreads();
	// Step 4: Compute dQ = dS * K / sqrt(D)
	const float scale = rsqrtf((float)D);
	for(int d_block = warp_id; d_block < (D + 15) / 16; d_block += num_warps){
		fill_fragment(c_frag, 0.0f);
		for(int col_block = 0; col_block < (T + 15) / 16; col_block++){
			// Load K tile
#pragma unroll
			for(int i = lane_id; i < 16 * 16; i += 32){
				int row = i / 16;
				int col = i % 16;
				if(col_block * 16 + row < T && d_block * 16 + col < D){ workspace_half[row * 16 + col] = K[batch_head_offset + (col_block * 16 + row) * D + d_block * 16 + col]; } else{ workspace_half[row * 16 + col] = __float2half(0.0f); }
			}
			__syncwarp();
			// Load dS fragment
			if(lane_id < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspace_half[256 + i * 16 + lane_id] = __float2half(col_block * 16 + lane_id < T ? dAtt_shared[i * T_padded + col_block * 16 + lane_id] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(a_frag, workspace_half + 256, 16);
			load_matrix_sync(b_frag, workspace_half, 16);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
		}
		// Scale and store to float workspace
#pragma unroll
		for(int i = 0; i < c_frag.num_elements; i++){ c_frag.x[i] *= scale; }
		store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
		__syncwarp();
		// Write to global dQ - convert float to half
#pragma unroll
		for(int i = lane_id; i < 16 * 16; i += 32){
			int row = i / 16;
			int col = i % 16;
			if(row_block * 16 + row < T && d_block * 16 + col < D){ atomicAdd(&dQ[batch_head_offset + (row_block * 16 + row) * D + d_block * 16 + col], __float2half(workspace_float[row * 16 + col])); }
		}
	}
	// Step 5: Compute dK = dS^T * Q / sqrt(D)
	for(int k_row_block = warp_id; k_row_block < (T + 15) / 16; k_row_block += num_warps){
		for(int d_block = 0; d_block < (D + 15) / 16; d_block++){
			fill_fragment(c_frag, 0.0f);
			// Prepare dS^T in workspace
			if(lane_id < 16){
#pragma unroll
				for(int i = 0; i < 16; i++){ workspace_half[lane_id * 16 + i] = __float2half(k_row_block * 16 + lane_id < T ? dAtt_shared[i * T_padded + k_row_block * 16 + lane_id] : 0.0f); }
			}
			__syncwarp();
			load_matrix_sync(a_frag, workspace_half, 16);
			load_matrix_sync(b_frag, Q_shared + d_block * 16, D_padded);
			mma_sync(c_frag, a_frag, b_frag, c_frag);
			// Scale
#pragma unroll
			for(int i = 0; i < c_frag.num_elements; i++){ c_frag.x[i] *= scale; }
			store_matrix_sync(workspace_float, c_frag, 16, mem_row_major);
			__syncwarp();
			// Write to global dK - convert float to half
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
	// Initialize gradients to zero
	cudaMemset(dQ, 0, B * H * T * D * sizeof(__half));
	cudaMemset(dK, 0, B * H * T * D * sizeof(__half));
	cudaMemset(dV, 0, B * H * T * D * sizeof(__half));
	// Use 128 threads (4 warps) for Volta
	dim3 block(128);
	dim3 grid((T + 15) / 16, B, H);
	// Calculate shared memory size with padding
	const int D_padded = (D + 7) & ~7;
	const int T_padded = (T + 3) & ~3;
	size_t shared_size = sizeof(__half) * (16 * D_padded * 4) + // dOut, V, K, Q tiles
		sizeof(float) * (16 * T_padded * 2) + // att_shared, dAtt_shared  
		sizeof(float) * 16 + // rowsum
		sizeof(__half) * 512 + // workspace_half
		sizeof(float) * 256; // workspace_float for WMMA accumulator storage
	// Configure for Volta's 96KB shared memory limit
	cudaFuncSetAttribute(WmmaAttentionBackwardKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	// Add L1 cache configuration for Volta
	cudaFuncSetAttribute(WmmaAttentionBackwardKernel, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
	WmmaAttentionBackwardKernel<<<grid, block, shared_size>>>(Q, K, V, dOut, AttentionWeights, dQ, dK, dV, B, T, D, H);
}