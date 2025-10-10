#include "CuCommon.cuh"
#define __CUDACC__
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
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
__global__ void ComputeDAttDQKernel(const __half* __restrict__ Q,
	const __half* __restrict__ K,
	const __half* __restrict__ V,
	const __half* __restrict__ dOut,
	const float* __restrict__ attention,
	float* __restrict__ dAtt,
	__half* __restrict__ dQ,
	int B, int T, int D, int H){
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
	extern __shared__ char sharedBytes[];
	float* attRows = reinterpret_cast<float*>(sharedBytes);
	float* dAttRows = attRows + 16 * T;
	float* rowSums = dAttRows + 16 * T;
	__half* tileA = reinterpret_cast<__half*>(rowSums + 16);
	__half* tileB = tileA + 16 * 16;
	float* fragStore = reinterpret_cast<float*>(tileB + 16 * 16);
	for(int idx = threadIdx.x; idx < 16 * T; idx += blockDim.x){
		const int localRow = idx / T;
		const int col = idx % T;
		const int globalRow = rowStart + localRow;
		float val = 0.0f;
		if(globalRow < T){ val = attention[attOffset + globalRow * T + col]; }
		attRows[localRow * T + col] = val;
		dAttRows[localRow * T + col] = 0.0f;
	}
	if(threadIdx.x < 16){ rowSums[threadIdx.x] = 0.0f; }
	__syncthreads();
	for(int keyBlock = 0; keyBlock < numKeyBlocks; ++keyBlock){
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> accFrag;
		fill_fragment(accFrag, 0.0f);
		for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = rowStart + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalRow < T && globalCol < D){ val = dOut[embOffset + globalRow * D + globalCol]; }
				tileA[r * 16 + c] = val;
			}
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = keyBlock * 16 + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalRow < T && globalCol < D){ val = V[embOffset + globalRow * D + globalCol]; }
				tileB[c * 16 + r] = val;
			}
			__syncthreads();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> outFrag;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> vFrag;
			load_matrix_sync(outFrag, tileA, 16);
			load_matrix_sync(vFrag, tileB, 16);
			mma_sync(accFrag, outFrag, vFrag, accFrag);
			__syncthreads();
		}
		store_matrix_sync(fragStore, accFrag, 16, wmma::mem_row_major);
		__syncthreads();
		for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalCol = keyBlock * 16 + c;
			if(rowStart + r < T && globalCol < T){
				dAttRows[r * T + globalCol] = fragStore[r * 16 + c];
			}
		}
		__syncthreads();
	}
	for(int row = threadIdx.x; row < 16; row += blockDim.x){
		const int globalRow = rowStart + row;
		float sum = 0.0f;
		if(globalRow < T){
			for(int col = 0; col < T; ++col){ sum += dAttRows[row * T + col] * attRows[row * T + col]; }
		}
		rowSums[row] = sum;
	}
	__syncthreads();
	for(int idx = threadIdx.x; idx < 16 * T; idx += blockDim.x){
		const int localRow = idx / T;
		const int col = idx % T;
		const int globalRow = rowStart + localRow;
		if(globalRow < T){
			const float a = attRows[localRow * T + col];
			const float raw = dAttRows[localRow * T + col];
			dAttRows[localRow * T + col] = a * (raw - rowSums[localRow]);
		} else{
			dAttRows[localRow * T + col] = 0.0f;
		}
	}
	__syncthreads();
	for(int idx = threadIdx.x; idx < 16 * T; idx += blockDim.x){
		const int localRow = idx / T;
		const int col = idx % T;
		const int globalRow = rowStart + localRow;
		if(globalRow < T){ dAtt[attOffset + globalRow * T + col] = dAttRows[localRow * T + col]; }
	}
	__syncthreads();
	const float scale = rsqrtf(static_cast<float>(D));
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> accFrag;
		fill_fragment(accFrag, 0.0f);
		for(int keyBlock = 0; keyBlock < numKeyBlocks; ++keyBlock){
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = rowStart + r;
				const int globalCol = keyBlock * 16 + c;
				float val = 0.0f;
				if(globalRow < T && globalCol < T){ val = dAttRows[r * T + globalCol]; }
				tileA[r * 16 + c] = __float2half(val);
			}
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalRow = keyBlock * 16 + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalRow < T && globalCol < D){ val = K[embOffset + globalRow * D + globalCol]; }
				tileB[r * 16 + c] = val;
			}
			__syncthreads();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> attFrag;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> kFrag;
			load_matrix_sync(attFrag, tileA, 16);
			load_matrix_sync(kFrag, tileB, 16);
			mma_sync(accFrag, attFrag, kFrag, accFrag);
			__syncthreads();
		}
		store_matrix_sync(fragStore, accFrag, 16, wmma::mem_row_major);
		__syncthreads();
		for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = rowStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < T && globalCol < D){
				const float val = fragStore[r * 16 + c] * scale;
				dQ[embOffset + globalRow * D + globalCol] = __float2half(val);
			}
		}
		__syncthreads();
	}
}

__global__ void ComputeDVKernel(const float* __restrict__ attention,
	const __half* __restrict__ dOut,
	__half* __restrict__ dV,
	int B, int T, int D, int H){
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
	extern __shared__ char sharedBytes[];
	float* fragStore = reinterpret_cast<float*>(sharedBytes);
	__half* tileA = reinterpret_cast<__half*>(fragStore + 16 * 16);
	__half* tileB = tileA + 16 * 16;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> accFrag;
		fill_fragment(accFrag, 0.0f);
		for(int rowBlock = 0; rowBlock < numRowBlocks; ++rowBlock){
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = rowBlock * 16 + r;
				const int globalKey = keyStart + c;
				float val = 0.0f;
				if(globalQuery < T && globalKey < T){ val = attention[attOffset + globalQuery * T + globalKey]; }
				tileA[c * 16 + r] = __float2half(val);
			}
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = rowBlock * 16 + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalQuery < T && globalCol < D){ val = dOut[embOffset + globalQuery * D + globalCol]; }
				tileB[r * 16 + c] = val;
			}
			__syncthreads();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> outFrag;
			load_matrix_sync(attFrag, tileA, 16);
			load_matrix_sync(outFrag, tileB, 16);
			mma_sync(accFrag, attFrag, outFrag, accFrag);
			__syncthreads();
		}
		store_matrix_sync(fragStore, accFrag, 16, wmma::mem_row_major);
		__syncthreads();
		for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = keyStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < T && globalCol < D){
				dV[embOffset + globalRow * D + globalCol] = __float2half(fragStore[r * 16 + c]);
			}
		}
		__syncthreads();
	}
}

__global__ void ComputeDKKernel(const float* __restrict__ dAtt,
	const __half* __restrict__ Q,
	__half* __restrict__ dK,
	int B, int T, int D, int H){
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
	extern __shared__ char sharedBytes[];
	float* fragStore = reinterpret_cast<float*>(sharedBytes);
	__half* tileA = reinterpret_cast<__half*>(fragStore + 16 * 16);
	__half* tileB = tileA + 16 * 16;
	for(int dBlock = 0; dBlock < numDBlocks; ++dBlock){
		wmma::fragment<wmma::accumulator, 16, 16, 16, float> accFrag;
		fill_fragment(accFrag, 0.0f);
		for(int rowBlock = 0; rowBlock < numRowBlocks; ++rowBlock){
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalKey = keyStart + r;
				const int globalQuery = rowBlock * 16 + c;
				float val = 0.0f;
				if(globalKey < T && globalQuery < T){ val = dAtt[attOffset + globalQuery * T + globalKey]; }
				tileA[c * 16 + r] = __float2half(val);
			}
			for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
				const int r = idx / 16;
				const int c = idx % 16;
				const int globalQuery = rowBlock * 16 + r;
				const int globalCol = dBlock * 16 + c;
				__half val = __float2half(0.0f);
				if(globalQuery < T && globalCol < D){ val = Q[embOffset + globalQuery * D + globalCol]; }
				tileB[r * 16 + c] = val;
			}
			__syncthreads();
			wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> attFrag;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> qFrag;
			load_matrix_sync(attFrag, tileA, 16);
			load_matrix_sync(qFrag, tileB, 16);
			mma_sync(accFrag, attFrag, qFrag, accFrag);
			__syncthreads();
		}
		store_matrix_sync(fragStore, accFrag, 16, wmma::mem_row_major);
		__syncthreads();
		for(int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x){
			const int r = idx / 16;
			const int c = idx % 16;
			const int globalRow = keyStart + r;
			const int globalCol = dBlock * 16 + c;
			if(globalRow < T && globalCol < D){
				const float val = fragStore[r * 16 + c] * scale;
				dK[embOffset + globalRow * D + globalCol] = __float2half(val);
			}
		}
		__syncthreads();
	}
}

void WmmaAttentionBackward(const __half* Q, const __half* K, const __half* V, const __half* dOut, const float* Att, __half* dQ, __half* dK, __half* dV, int B, int T, int D, int H){
	const int numRowBlocks = (T + 15) / 16;
	const int numKeyBlocks = (T + 15) / 16;
	dim3 block(256);
	dim3 gridDQ(numRowBlocks, B, H);
	dim3 gridKV(numKeyBlocks, B, H);
	size_t smemDQ = sizeof(float) * (16 * T * 2 + 16 + 16 * 16) + sizeof(__half) * (16 * 16 * 2);
	size_t smemDV = sizeof(float) * 16 * 16 + sizeof(__half) * (16 * 16 * 2);
	float* dAttWorkspace = nullptr;
	cudaError_t err = cudaMalloc(&dAttWorkspace, sizeof(float) * B * H * T * T);
	if(err != cudaSuccess){
		printf("WmmaAttention Backward workspace alloc error: %s\n", cudaGetErrorString(err));
		return;
	}
	cudaFuncSetAttribute(ComputeDAttDQKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	cudaFuncSetAttribute(ComputeDVKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	cudaFuncSetAttribute(ComputeDKKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
	ComputeDAttDQKernel<<<gridDQ, block, smemDQ>>>(Q, K, V, dOut, Att, dAttWorkspace, dQ, B, T, D, H);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttention Backward dAtt+dQ error: %s\n", cudaGetErrorString(err));
		cudaFree(dAttWorkspace);
		return;
	}
	ComputeDVKernel<<<gridKV, block, smemDV>>>(Att, dOut, dV, B, T, D, H);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttention Backward dV error: %s\n", cudaGetErrorString(err));
		cudaFree(dAttWorkspace);
		return;
	}
	ComputeDKKernel<<<gridKV, block, smemDV>>>(dAttWorkspace, Q, dK, B, T, D, H);
	err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("WmmaAttention Backward dK error: %s\n", cudaGetErrorString(err));
		cudaFree(dAttWorkspace);
		return;
	}
	cudaFree(dAttWorkspace);
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess) printf("WmmaAttention Backward sync error: %s\n", cudaGetErrorString(err));
}