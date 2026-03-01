#define __CUDACC__
#include "CuCommon.cuh"
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdint>
using namespace nvcuda;
static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int WMMA_K = 16;
template <bool Trans, int TK> __device__ __forceinline__ void LoadATile(__half* __restrict__ dst, const __half* __restrict__ src, int rowBase, int kk, int lda, int m, int k, int tid, bool fullM, bool fullK, bool vec){
	constexpr int ELM = TILE_M * TK;
	constexpr int VEC = ELM / 2;   
	if(!Trans){
		constexpr int S = TILE_M + 8;
		if(fullM && fullK && vec){
			for(int idx = tid; idx < VEC; idx += 256){
				int rp = idx & 31;
				int c = idx >> 5;
				int r = rp << 1;
				*reinterpret_cast<__half2*>(dst + r + c * S) = *reinterpret_cast<const __half2*>(src + (rowBase + r) + static_cast<long long>(kk + c) * lda);
			}
		} else{
			for(int idx = tid; idx < ELM; idx += 256){
				int r = idx & 63;
				int c = idx >> 6;
				int gr = rowBase + r, gc = kk + c;
				dst[r + c * S] = (gr < m && gc < k) ? src[gr + static_cast<long long>(gc) * lda] : __float2half(0.f);
			}
		}
	} else{
		constexpr int S = TK + 8;
		if(fullM && fullK && vec){
			for(int idx = tid; idx < VEC; idx += 256){
				int r = idx & 63;
				int cp = idx >> 6;
				int c = cp << 1;
				*reinterpret_cast<__half2*>(dst + r * S + c) = *reinterpret_cast<const __half2*>(src + (kk + c) + static_cast<long long>(rowBase + r) * lda);
			}
		} else{
			for(int idx = tid; idx < ELM; idx += 256){
				int r = idx & 63;
				int c = idx >> 6;
				int gr = kk + c, gc = rowBase + r;
				dst[r * S + c] = (gc < m && gr < k) ? src[gr + static_cast<long long>(gc) * lda] : __float2half(0.f);
			}
		}
	}
}
template <bool Trans, int TK> __device__ __forceinline__ void LoadBTile(__half* __restrict__ dst, const __half* __restrict__ src, int colBase, int kk, int ldb, int n, int k, int tid, bool fullN, bool fullK, bool vec){
	constexpr int S = TK + 8;
	constexpr int ELM = TK * TILE_N;
	constexpr int VEC = ELM / 2;
	constexpr int HTK = TK / 2;     
	constexpr int HN = TILE_N / 2;
	if(!Trans){
		if(fullN && fullK && vec){
			for(int idx = tid; idx < VEC; idx += 256){
				int rp = idx & (HTK - 1);     
				int c = idx / HTK;      
				int r = rp << 1;
				*reinterpret_cast<__half2*>(dst + r + c * S) = *reinterpret_cast<const __half2*>(src + (kk + r) + static_cast<long long>(colBase + c) * ldb);
			}
		} else{
			for(int idx = tid; idx < ELM; idx += 256){
				int r = idx & (TK - 1);
				int c = idx / TK;
				int gr = kk + r, gc = colBase + c;
				dst[r + c * S] = (gr < k && gc < n) ? src[gr + static_cast<long long>(gc) * ldb] : __float2half(0.f);
			}
		}
	} else{
		if(fullN && fullK && vec){
			for(int idx = tid; idx < VEC; idx += 256){
				int r = idx / HN;   
				int cp = idx & (HN - 1);   
				int c = cp << 1;
				__half2 v = *reinterpret_cast<const __half2*>(src + (colBase + c) + static_cast<long long>(kk + r) * ldb);
				dst[r + c * S] = __low2half(v);
				dst[r + (c + 1) * S] = __high2half(v);
			}
		} else{
			for(int idx = tid; idx < ELM; idx += 256){
				int r = idx & (TK - 1);
				int c = idx / TK;
				int gn = colBase + c, gk = kk + r;
				dst[r + c * S] = (gn < n && gk < k) ? src[gn + static_cast<long long>(gk) * ldb] : __float2half(0.f);
			}
		}
	}
}
__global__ void PreScaleCKernel(__half* __restrict__ C, int m, int n, int ldc, float beta, long long strideC, int batchCount){
	const int batchElems = m * n;
	const int total = batchElems * batchCount;
	for(int i = blockIdx.x * 256 + threadIdx.x; i < total; i += gridDim.x * 256){
		int b = i / batchElems;
		int local = i - b * batchElems;
		int r = local % m;
		int c = local / m;
		__half* bC = C + b * strideC;
		long long a = r + static_cast<long long>(c) * ldc;
		bC[a] = (beta == 0.f) ? __float2half(0.f) : __float2half(beta * __half2float(bC[a]));
	}
}
template <bool TransA, bool TransB, int TK, int MIN_BLK> __global__ __launch_bounds__(256, MIN_BLK)void HgemmWmma64x64(const __half* __restrict__ A, const __half* __restrict__ B, __half* __restrict__ C, const int m, const int n, const int k, const int lda, const int ldb, const int ldc,
																														const float alpha, const float beta, const long long strideA, const long long strideB, const long long strideC, const int splitK){
	constexpr int LDA_SM = TransA ? (TK + 8) : (TILE_M + 8);
	constexpr int LDB_SM = TK + 8;
	constexpr int A_EL = TransA ? (TILE_M * LDA_SM) : (TK * LDA_SM);
	constexpr int B_EL = TILE_N * LDB_SM;
	constexpr int BUF2B = (A_EL + B_EL) * 2 * 2;   
	constexpr int C_B = TILE_M * TILE_N * static_cast<int>(sizeof(float));
	constexpr int SMEM = (BUF2B > C_B) ? BUF2B : C_B;
	__shared__ __align__(32) char smem_raw[SMEM];
	__half *aBuf[2], *bBuf[2];
	aBuf[0] = reinterpret_cast<__half*>(smem_raw);
	bBuf[0] = aBuf[0] + A_EL;
	aBuf[1] = bBuf[0] + B_EL;
	bBuf[1] = aBuf[1] + A_EL;
	auto cBuf = reinterpret_cast<float*>(smem_raw);
	const int splitIdx = blockIdx.z % splitK;
	const int batch = blockIdx.z / splitK;
	const int kPerSplit = (k + splitK - 1) / splitK;
	const int kStart = splitIdx * kPerSplit;
	const int kEnd = min(kStart + kPerSplit, k);
	const int kSlice = kEnd - kStart;
	const int rowBase = static_cast<int>(blockIdx.x) * TILE_M;
	const int colBase = static_cast<int>(blockIdx.y) * TILE_N;
	if(rowBase >= m || colBase >= n || kSlice <= 0) return;
	const __half* bA = A + batch * strideA;
	const __half* bB = B + batch * strideB;
	__half* bC = C + batch * strideC;
	const int tid = threadIdx.x;
	const int warpId = tid >> 5;
	const int warpRow = warpId & 3;
	const int warpCol = warpId >> 2;
	const bool fullM = (rowBase + TILE_M <= m);
	const bool fullN = (colBase + TILE_N <= n);
	const bool vecA = !(reinterpret_cast<uintptr_t>(bA) & 3u) && !(lda & 1);
	const bool vecB = !(reinterpret_cast<uintptr_t>(bB) & 3u) && !(ldb & 1);
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0, acc1;
	fill_fragment(acc0, 0.f);
	fill_fragment(acc1, 0.f);
	const int numChunks = (kSlice + TK - 1) / TK;
	{
		bool fk = (TK <= kSlice);
		LoadATile<TransA, TK>(aBuf[0], bA, rowBase, kStart, lda, m, k, tid, fullM, fk, vecA);
		LoadBTile<TransB, TK>(bBuf[0], bB, colBase, kStart, ldb, n, k, tid, fullN, fk, vecB);
	}
	__syncthreads();
	for(int ti = 0; ti < numChunks; ++ti){
		const int cur = ti & 1;
		const int nxt = 1 - cur;
		if(ti + 1 < numChunks){
			int nkk = kStart + (ti + 1) * TK;
			bool fkn = (nkk + TK <= kEnd);
			LoadATile<TransA, TK>(aBuf[nxt], bA, rowBase, nkk, lda, m, k, tid, fullM, fkn, vecA);
			LoadBTile<TransB, TK>(bBuf[nxt], bB, colBase, nkk, ldb, n, k, tid, fullN, fkn, vecB);
		}
		const int bc = warpCol * 32;
#pragma unroll
		for(int ki = 0; ki < TK / WMMA_K; ++ki){
			const int kOff = ki * WMMA_K;
			wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bF0, bF1;
			load_matrix_sync(bF0, bBuf[cur] + kOff + bc * LDB_SM, LDB_SM);
			load_matrix_sync(bF1, bBuf[cur] + kOff + (bc + 16) * LDB_SM, LDB_SM);
			if(!TransA){
				wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> aF;
				load_matrix_sync(aF, aBuf[cur] + warpRow * 16 + kOff * LDA_SM, LDA_SM);
				mma_sync(acc0, aF, bF0, acc0);
				mma_sync(acc1, aF, bF1, acc1);
			} else{
				wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aF;
				load_matrix_sync(aF, aBuf[cur] + warpRow * 16 * LDA_SM + kOff, LDA_SM);
				mma_sync(acc0, aF, bF0, acc0);
				mma_sync(acc1, aF, bF1, acc1);
			}
		}
		if(ti + 1 < numChunks) __syncthreads();
	}
	__syncthreads();
	{
		const int bc = warpCol * 32;
		store_matrix_sync(cBuf + warpRow * 16 + bc * TILE_M, acc0, TILE_M, wmma::mem_col_major);
		store_matrix_sync(cBuf + warpRow * 16 + (bc + 16) * TILE_M, acc1, TILE_M, wmma::mem_col_major);
	}
	__syncthreads();
	const bool vecC = !(reinterpret_cast<uintptr_t>(bC) & 3u) && !(ldc & 1);
	const bool fullC = fullM && fullN;
	if(splitK <= 1){
		if(fullC && vecC){
			if(beta == 0.f){
				for(int idx = tid; idx < 2048; idx += 256){
					int rp = idx & 31, c = idx >> 5, r = rp << 1;
					float v0 = cBuf[r + c * TILE_M];
					float v1 = cBuf[r + 1 + c * TILE_M];
					*reinterpret_cast<__half2*>(bC + (rowBase + r) + static_cast<long long>(colBase + c) * ldc) = __floats2half2_rn(alpha * v0, alpha * v1);
				}
			} else{
				for(int idx = tid; idx < 2048; idx += 256){
					int rp = idx & 31, c = idx >> 5, r = rp << 1;
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
			if(beta == 0.f){
				for(int idx = tid; idx < TILE_M * TILE_N; idx += 256){
					int r = idx & (TILE_M - 1), c = idx >> 6;
					int gr = rowBase + r, gc = colBase + c;
					if(gr < m && gc < n)
						bC[gr + static_cast<long long>(gc) * ldc] = __float2half(alpha * cBuf[r + c * TILE_M]);
				}
			} else{
				for(int idx = tid; idx < TILE_M * TILE_N; idx += 256){
					int r = idx & (TILE_M - 1), c = idx >> 6;
					int gr = rowBase + r, gc = colBase + c;
					if(gr < m && gc < n){
						long long ga = gr + static_cast<long long>(gc) * ldc;
						float p = __half2float(bC[ga]);
						bC[ga] = __float2half(alpha * cBuf[r + c * TILE_M] + beta * p);
					}
				}
			}
		}
	} else{
		for(int idx = tid; idx < TILE_M * TILE_N; idx += 256){
			int r = idx & (TILE_M - 1), c = idx >> 6;
			int gr = rowBase + r, gc = colBase + c;
			if(gr < m && gc < n){
				float v = alpha * cBuf[r + c * TILE_M];
				atomicAdd(&bC[gr + static_cast<long long>(gc) * ldc], __float2half(v));
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
static float ReadAlpha(const void* a){ return a ? *static_cast<const float*>(a) : 1.f; }
static float ReadBeta(const void* b){ return b ? *static_cast<const float*>(b) : 0.f; }
static int ChooseSplitK(int m, int n, int k, int batchCount){
	int mnTiles = DivCeil(m, TILE_M) * DivCeil(n, TILE_N);
	int totalTiles = mnTiles * batchCount;
	constexpr int TARGET = 160;      
	if(totalTiles >= TARGET) return 1;
	int splitK = (TARGET + totalTiles - 1) / totalTiles;
	splitK = min(splitK, max(1, k / 256));       
	splitK = min(splitK, 32);
	return max(splitK, 1);
}
template <bool TA, bool TB> static void LaunchGemm(const __half* A, const __half* B, __half* C, int m, int n, int k, int lda, int ldb, int ldc, float alpha, float beta, long long sA, long long sB, long long sC, int batchCount, cudaStream_t stream){
	const int mnTiles = DivCeil(m, TILE_M) * DivCeil(n, TILE_N);
	const int totalTiles = mnTiles * batchCount;
	constexpr int SM_TARGET = 160;
	const bool useWideK = (k >= 512) && (totalTiles < SM_TARGET);
	if(useWideK){
		int splitK = ChooseSplitK(m, n, k, batchCount);
		if(splitK > 1){
			int elems = m * n * batchCount;
			PreScaleCKernel<<<min(DivCeil(elems, 256), 1024), 256, 0, stream>>>(C, m, n, ldc, beta, sC, batchCount);
		}
		const dim3 grid(DivCeil(m, TILE_M), DivCeil(n, TILE_N), splitK * batchCount);
		float kernelBeta = (splitK > 1) ? 0.f : beta;
		HgemmWmma64x64<TA, TB, 32, 2><<<grid, 256, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, alpha, kernelBeta, sA, sB, sC, splitK);
	} else{
		const dim3 grid(DivCeil(m, TILE_M), DivCeil(n, TILE_N), batchCount);
		HgemmWmma64x64<TA, TB, 16, 3><<<grid, 256, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, alpha, beta, sA, sB, sC, 1);
	}
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