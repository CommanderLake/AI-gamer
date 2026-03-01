#define __CUDACC__
#include "CuCommon.cuh"
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdint>
using namespace nvcuda;
template <bool Trans> __device__ __forceinline__ void LoadATile(__half* __restrict__ dst, const __half* __restrict__ src, int rowBase, int kk, int lda, int m, int k, int tid, bool fullM, bool fullK, bool vec){
	const int TILE_M = 64;
	const int TILE_K = 16;
	if(!Trans){
		const int S = TILE_M + 8;
		if(fullM && fullK && vec){
			for(int idx = tid; idx < 512; idx += 256){
				int rp = idx & 31;
				int c = idx >> 5;
				int r = rp << 1;
				*reinterpret_cast<__half2*>(dst + r + c * S) = *reinterpret_cast<const __half2*>(src + (rowBase + r) + static_cast<long long>(kk + c) * lda);
			}
		} else{
			for(int idx = tid; idx < 1024; idx += 256){
				int r = idx & 63;
				int c = idx >> 6;
				int gr = rowBase + r;
				int gc = kk + c;
				dst[r + c * S] = (gr < m && gc < k) ? src[gr + static_cast<long long>(gc) * lda] : __float2half(0.0f);
			}
		}
	} else{
		const int S = TILE_K + 8;
		if(fullM && fullK && vec){
			for(int idx = tid; idx < 512; idx += 256){
				int r = idx & 63;
				int cp = idx >> 6;
				int c = cp << 1;
				*reinterpret_cast<__half2*>(dst + r * S + c) = *reinterpret_cast<const __half2*>(src + (kk + c) + static_cast<long long>(rowBase + r) * lda);
			}
		} else{
			for(int idx = tid; idx < 1024; idx += 256){
				int r = idx & 63;
				int c = idx >> 6;
				int gr = kk + c;
				int gc = rowBase + r;
				dst[r * S + c] = (gc < m && gr < k) ? src[gr + static_cast<long long>(gc) * lda] : __float2half(0.0f);
			}
		}
	}
}
template <bool Trans> __device__ __forceinline__ void LoadBTile(__half* __restrict__ dst, const __half* __restrict__ src, int colBase, int kk, int ldb, int n, int k, int tid, bool fullN, bool fullK, bool vec){
	const int TILE_K = 16;
	const int S = TILE_K + 8;
	if(!Trans){
		if(fullN && fullK && vec){
			for(int idx = tid; idx < 512; idx += 256){
				int rp = idx & 7;
				int c = idx >> 3;
				int r = rp << 1;
				*reinterpret_cast<__half2*>(dst + r + c * S) = *reinterpret_cast<const __half2*>(src + (kk + r) + static_cast<long long>(colBase + c) * ldb);
			}
		} else{
			for(int idx = tid; idx < 1024; idx += 256){
				int r = idx & 15;
				int c = idx >> 4;
				int gr = kk + r;
				int gc = colBase + c;
				dst[r + c * S] = (gr < k && gc < n) ? src[gr + static_cast<long long>(gc) * ldb] : __float2half(0.0f);
			}
		}
	} else{
		if(fullN && fullK && vec){
			for(int idx = tid; idx < 512; idx += 256){
				int r = idx >> 5;
				int cp = idx & 31;
				int c = cp << 1;
				__half2 v = *reinterpret_cast<const __half2*>(src + (colBase + c) + static_cast<long long>(kk + r) * ldb);
				dst[r + c * S] = __low2half(v);
				dst[r + (c + 1) * S] = __high2half(v);
			}
		} else{
			for(int idx = tid; idx < 1024; idx += 256){
				int r = idx & 15;
				int c = idx >> 4;
				int gn = colBase + c;
				int gk = kk + r;
				dst[r + c * S] = (gn < n && gk < k) ? src[gn + static_cast<long long>(gk) * ldb] : __float2half(0.0f);
			}
		}
	}
}
template <bool TransA, bool TransB> __global__ __launch_bounds__(256, 3) void HgemmWmma64x64(const __half* __restrict__ A, const __half* __restrict__ B, __half* __restrict__ C, const int m, const int n, const int k, const int lda, const int ldb, const int ldc, const float alpha, const float beta, const long long strideA, const long long strideB, const long long strideC){
	const int TILE_M = 64;
	const int TILE_N = 64;
	const int TILE_K = 16;
	const int LDA_SM = TransA ? (TILE_K + 8) : (TILE_M + 8);
	const int LDB_SM = TILE_K + 8;
	const int A_EL = TransA ? (TILE_M * LDA_SM) : (TILE_K * LDA_SM);
	const int B_EL = TILE_N * LDB_SM;
	__shared__ __align__(32) char smem_raw[16384];
	__half* aBuf[2];
	__half* bBuf[2];
	aBuf[0] = reinterpret_cast<__half*>(smem_raw);
	bBuf[0] = aBuf[0] + A_EL;
	aBuf[1] = bBuf[0] + B_EL;
	bBuf[1] = aBuf[1] + A_EL;
	auto cBuf = reinterpret_cast<float*>(smem_raw);
	const int batch = blockIdx.z;
	const int rowBase = static_cast<int>(blockIdx.x) * TILE_M;
	const int colBase = static_cast<int>(blockIdx.y) * TILE_N;
	if(rowBase >= m || colBase >= n) return;
	const __half* bA = A + batch * strideA;
	const __half* bB = B + batch * strideB;
	__half* bC = C + batch * strideC;
	const int tid = static_cast<int>(threadIdx.x);
	const int warpId = tid >> 5;
	const int warpRow = warpId & 3;
	const int warpCol = warpId >> 2;
	const bool fullM = (rowBase + TILE_M <= m);
	const bool fullN = (colBase + TILE_N <= n);
	const bool vecA = ((reinterpret_cast<uintptr_t>(bA) & 3u) == 0u) && ((lda & 1) == 0);
	const bool vecB = ((reinterpret_cast<uintptr_t>(bB) & 3u) == 0u) && ((ldb & 1) == 0);
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
	fill_fragment(acc0, 0.0f);
	fill_fragment(acc1, 0.0f);
	const int numK = (k + TILE_K - 1) / TILE_K;
	{
		bool fk = (TILE_K <= k);
		LoadATile<TransA>(aBuf[0], bA, rowBase, 0, lda, m, k, tid, fullM, fk, vecA);
		LoadBTile<TransB>(bBuf[0], bB, colBase, 0, ldb, n, k, tid, fullN, fk, vecB);
	}
	__syncthreads();
	for(int ti = 0; ti < numK; ++ti){
		const int cur = ti & 1;
		const int nxt = 1 - cur;
		if(ti + 1 < numK){
			int nextKK = (ti + 1) * TILE_K;
			bool fkn = (nextKK + TILE_K <= k);
			LoadATile<TransA>(aBuf[nxt], bA, rowBase, nextKK, lda, m, k, tid, fullM, fkn, vecA);
			LoadBTile<TransB>(bBuf[nxt], bB, colBase, nextKK, ldb, n, k, tid, fullN, fkn, vecB);
		}
		{
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag0, bFrag1;
			const int bc = warpCol * 32;
			load_matrix_sync(bFrag0, bBuf[cur] + bc * LDB_SM, LDB_SM);
			load_matrix_sync(bFrag1, bBuf[cur] + (bc + 16) * LDB_SM, LDB_SM);
			if(!TransA){
				wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> aFrag;
				load_matrix_sync(aFrag, aBuf[cur] + warpRow * 16, LDA_SM);
				mma_sync(acc0, aFrag, bFrag0, acc0);
				mma_sync(acc1, aFrag, bFrag1, acc1);
			} else{
				wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aFrag;
				load_matrix_sync(aFrag, aBuf[cur] + warpRow * 16 * LDA_SM, LDA_SM);
				mma_sync(acc0, aFrag, bFrag0, acc0);
				mma_sync(acc1, aFrag, bFrag1, acc1);
			}
		}
		if(ti + 1 < numK) __syncthreads();
	}
	__syncthreads();
	const int bc = warpCol * 32;
	store_matrix_sync(cBuf + warpRow * 16 + bc * TILE_M, acc0, TILE_M, wmma::mem_col_major);
	store_matrix_sync(cBuf + warpRow * 16 + (bc + 16) * TILE_M, acc1, TILE_M, wmma::mem_col_major);
	__syncthreads();
	const bool vecC = ((reinterpret_cast<uintptr_t>(bC) & 3u) == 0u) && ((ldc & 1) == 0);
	const bool fullC = fullM && fullN;
	if(fullC && vecC){
		if(beta == 0.0f){
			for(int idx = tid; idx < 2048; idx += 256){
				int rp = idx & 31;
				int c = idx >> 5;
				int r = rp << 1;
				float v0 = cBuf[r + c * TILE_M];
				float v1 = cBuf[r + 1 + c * TILE_M];
				*reinterpret_cast<__half2*>(bC + (rowBase + r) + static_cast<long long>(colBase + c) * ldc) = __floats2half2_rn(alpha * v0, alpha * v1);
			}
		} else{
			for(int idx = tid; idx < 2048; idx += 256){
				int rp = idx & 31;
				int c = idx >> 5;
				int r = rp << 1;
				float v0 = cBuf[r + c * TILE_M];
				float v1 = cBuf[r + 1 + c * TILE_M];
				long long ga = (rowBase + r) + static_cast<long long>(colBase + c) * ldc;
				__half2 prev = *reinterpret_cast<const __half2*>(bC + ga);
				float p0 = __half2float(__low2half(prev));
				float p1 = __half2float(__high2half(prev));
				*reinterpret_cast<__half2*>(bC + ga) = __floats2half2_rn(alpha * v0 + beta * p0, alpha * v1 + beta * p1);
			}
		}
	} else{
		if(beta == 0.0f){
			for(int idx = tid; idx < TILE_M * TILE_N; idx += 256){
				int r = idx & (TILE_M - 1);
				int c = idx >> 6;
				int gr = rowBase + r;
				int gc = colBase + c;
				if(gr < m && gc < n)
					bC[gr + static_cast<long long>(gc) * ldc] = __float2half(alpha * cBuf[r + c * TILE_M]);
			}
		} else{
			for(int idx = tid; idx < TILE_M * TILE_N; idx += 256){
				int r = idx & (TILE_M - 1);
				int c = idx >> 6;
				int gr = rowBase + r;
				int gc = colBase + c;
				if(gr < m && gc < n){
					long long ga = gr + static_cast<long long>(gc) * ldc;
					float p = __half2float(bC[ga]);
					bC[ga] = __float2half(alpha * cBuf[r + c * TILE_M] + beta * p);
				}
			}
		}
	}
}
bool CanUseWmma(cudaDataType Atype, cudaDataType Btype, cudaDataType Ctype, cudaDataType computeType, CLNNOpT transa, CLNNOpT transb){
	if(Atype != CUDA_R_16F || Btype != CUDA_R_16F || Ctype != CUDA_R_16F) return false;
	if(computeType != CUDA_R_32F) return false;
	if((transa != CLNN_OP_N && transa != CLNN_OP_T) || (transb != CLNN_OP_N && transb != CLNN_OP_T)) return false;
	cudaDeviceProp prop{};
	if(cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
	return prop.major == 7;
}
static float ReadAlpha(const void* a){ return a ? *static_cast<const float*>(a) : 1.0f; }
static float ReadBeta(const void* b){ return b ? *static_cast<const float*>(b) : 0.0f; }
template <bool TA, bool TB> static void LaunchGemm(const __half* A, const __half* B, __half* C, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, long long sA, long long sB, long long sC, int batchCount, cudaStream_t stream){
	const dim3 block(256);
	const dim3 grid(DivCeil(m, 64), DivCeil(n, 64), batchCount);
	HgemmWmma64x64<TA, TB><<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, alpha, beta, sA, sB, sC);
}
static CLNNStatusT DispatchGemm(CLNNOpT ta, CLNNOpT tb, const __half* A, const __half* B, __half* C, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, long long sA, long long sB, long long sC, int batchCount, cudaStream_t stream){
	if(ta == CLNN_OP_N && tb == CLNN_OP_N) LaunchGemm<false, false>(A, B, C, m, n, k, lda, ldb, ldc, alpha, beta, sA, sB, sC, batchCount, stream);
	else if(ta == CLNN_OP_T && tb == CLNN_OP_N) LaunchGemm<true, false>(A, B, C, m, n, k, lda, ldb, ldc, alpha, beta, sA, sB, sC, batchCount, stream);
	else if(ta == CLNN_OP_N && tb == CLNN_OP_T) LaunchGemm<false, true>(A, B, C, m, n, k, lda, ldb, ldc, alpha, beta, sA, sB, sC, batchCount, stream);
	else LaunchGemm<true, true>(A, B, C, m, n, k, lda, ldb, ldc, alpha, beta, sA, sB, sC, batchCount, stream);
	checkCUDA(cudaDeviceSynchronize());
	return (cudaPeekAtLastError() == cudaSuccess) ? CLNN_STATUS_SUCCESS : CLNN_STATUS_EXECUTION_FAILED;
}
CLNNStatusT CLNNGemmEx(CLNNOpT transa, CLNNOpT transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cudaDataType computeType){
	if(!CanUseWmma(Atype, Btype, Ctype, computeType, transa, transb)) return CLNN_STATUS_NOT_SUPPORTED;
	return DispatchGemm(transa, transb, static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, ReadAlpha(alpha), ReadBeta(beta), 0, 0, 0, 1, nullptr);
}
CLNNStatusT CLNNGemmStridedBatchedEx(CLNNOpT transa, CLNNOpT transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long sA, const void* B, cudaDataType Btype, int ldb, long long sB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long sC, int batchCount, cudaDataType computeType){
	if(batchCount <= 0) return CLNN_STATUS_INVALID_VALUE;
	if(!CanUseWmma(Atype, Btype, Ctype, computeType, transa, transb)) return CLNN_STATUS_NOT_SUPPORTED;
	return DispatchGemm(transa, transb, static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, ReadAlpha(alpha), ReadBeta(beta), sA, sB, sC, batchCount, nullptr);
}