#define __CUDACC__
#include "CuCommon.cuh"
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdint>
using namespace nvcuda;
__device__ __forceinline__ __half LoadHalfElement(const __half* ptr, const int row, const int col, const int ld){ return ptr[row + col * ld]; }
__device__ __forceinline__ __half GetOpA(const __half* A, const CLNNOpT transa, const int row, const int col, const int lda){ return transa == CLNN_OP_N ? LoadHalfElement(A, row, col, lda) : LoadHalfElement(A, col, row, lda); }
__device__ __forceinline__ __half GetOpB(const __half* B, const CLNNOpT transb, const int row, const int col, const int ldb){ return transb == CLNN_OP_N ? LoadHalfElement(B, row, col, ldb) : LoadHalfElement(B, col, row, ldb); }

__global__ __launch_bounds__(256) void GemmExWmmaKernelNN64(const __half* A, const __half* B, __half* C, const int m, const int n, const int k, const int lda, const int ldb, const int ldc, const float alpha, const float beta, const long long strideA, const long long strideB, const long long strideC){
	constexpr int SKEW_A = 8;
	constexpr int SKEW_B = 8;
	constexpr int LDM_A = 64 + SKEW_A;
	constexpr int LDM_B = 16 + SKEW_B;
	const int batch = blockIdx.z;
	const int rowBase = (int)blockIdx.x * 64;
	const int colBase = (int)blockIdx.y * 64;
	if(rowBase >= m || colBase >= n) return;
	const __half* batchA = A + batch * strideA;
	const __half* batchB = B + batch * strideB;
	__half* batchC = C + batch * strideC;
	const int tid = (int)threadIdx.x;
	const int warpId = tid >> 5;
	const int warpRow = warpId & 3;
	const int warpCol = warpId >> 2;
	__shared__ __align__(32) __half aTile[64 * LDM_A];
	__shared__ __align__(32) __half bTile[64 * LDM_B];
	__shared__ __align__(32) float cTile[64 * 64];
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;
	wmma::fill_fragment(acc0, 0.0f);
	wmma::fill_fragment(acc1, 0.0f);
	const bool vecA = (((uintptr_t)batchA & 3u) == 0u) && ((lda & 1) == 0);
	const bool vecB = (((uintptr_t)batchB & 3u) == 0u) && ((ldb & 1) == 0);
	const bool fullA = (rowBase + 63 < m);
	const bool fullB = (colBase + 63 < n);
	for(int kk = 0; kk < k; kk += 16){
		const bool fullK = (kk + 15 < k);
		if(vecA && fullA && fullK){
			for(int idx = tid; idx < 512; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = kk + c;
				const __half2 v = *reinterpret_cast<const __half2*>(batchA + row + col * lda);
				*reinterpret_cast<__half2*>(aTile + r0 + c * LDM_A) = v;
			}
		} else{
			for(int idx = tid; idx < 512; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row0 = rowBase + r0;
				const int row1 = row0 + 1;
				const int col = kk + c;
				aTile[r0 + c * LDM_A] = (row0 < m && col < k) ? batchA[row0 + col * lda] : __float2half(0.0f);
				aTile[r0 + 1 + c * LDM_A] = (row1 < m && col < k) ? batchA[row1 + col * lda] : __float2half(0.0f);
			}
		}
		if(vecB && fullB && fullK){
			for(int idx = tid; idx < 512; idx += 256){
				const int rp = idx & 7;
				const int c = idx >> 3;
				const int r0 = rp * 2;
				const int row = kk + r0;
				const int col = colBase + c;
				const __half2 v = *reinterpret_cast<const __half2*>(batchB + row + col * ldb);
				*reinterpret_cast<__half2*>(bTile + r0 + c * LDM_B) = v;
			}
		} else{
			for(int idx = tid; idx < 512; idx += 256){
				const int rp = idx & 7;
				const int c = idx >> 3;
				const int r0 = rp * 2;
				const int row0 = kk + r0;
				const int row1 = row0 + 1;
				const int col = colBase + c;
				bTile[r0 + c * LDM_B] = (row0 < k && col < n) ? batchB[row0 + col * ldb] : __float2half(0.0f);
				bTile[r0 + 1 + c * LDM_B] = (row1 < k && col < n) ? batchB[row1 + col * ldb] : __float2half(0.0f);
			}
		}
		__syncthreads();
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> aFrag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag0;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag1;
		const int baseCol = warpCol * 32;
		wmma::load_matrix_sync(aFrag, aTile + warpRow * 16, LDM_A);
		wmma::load_matrix_sync(bFrag0, bTile + (baseCol + 0) * LDM_B, LDM_B);
		wmma::load_matrix_sync(bFrag1, bTile + (baseCol + 16) * LDM_B, LDM_B);
		wmma::mma_sync(acc0, aFrag, bFrag0, acc0);
		wmma::mma_sync(acc1, aFrag, bFrag1, acc1);
		__syncthreads();
	}
	const int baseCol = warpCol * 32;
	wmma::store_matrix_sync(cTile + (warpRow * 16) + (baseCol + 0) * 64, acc0, 64, wmma::mem_col_major);
	wmma::store_matrix_sync(cTile + (warpRow * 16) + (baseCol + 16) * 64, acc1, 64, wmma::mem_col_major);
	__syncthreads();
	const bool vecC = (((uintptr_t)batchC & 3u) == 0u) && ((ldc & 1) == 0);
	const bool fullC = (rowBase + 63 < m) && (colBase + 63 < n);
	if(vecC && fullC){
		if(beta == 0.0f){
			for(int idx = tid; idx < 2048; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = colBase + c;
				const float v0 = cTile[r0 + c * 64];
				const float v1 = cTile[r0 + 1 + c * 64];
				*reinterpret_cast<__half2*>(batchC + row + col * ldc) = __floats2half2_rn(alpha * v0, alpha * v1);
			}
		} else{
			for(int idx = tid; idx < 2048; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = colBase + c;
				const float v0 = cTile[r0 + c * 64];
				const float v1 = cTile[r0 + 1 + c * 64];
				const __half2 prevh = *reinterpret_cast<const __half2*>(batchC + row + col * ldc);
				const float p0 = __half2float(__low2half(prevh));
				const float p1 = __half2float(__high2half(prevh));
				*reinterpret_cast<__half2*>(batchC + row + col * ldc) = __floats2half2_rn(alpha * v0 + beta * p0, alpha * v1 + beta * p1);
			}
		}
	} else{
		if(beta == 0.0f){
			for(int idx = tid; idx < 4096; idx += 256){
				const int r = idx & 63;
				const int c = idx >> 6;
				const int row = rowBase + r;
				const int col = colBase + c;
				if(row < m && col < n) batchC[row + col * ldc] = __float2half(alpha * cTile[r + c * 64]);
			}
		} else{
			for(int idx = tid; idx < 4096; idx += 256){
				const int r = idx & 63;
				const int c = idx >> 6;
				const int row = rowBase + r;
				const int col = colBase + c;
				if(row < m && col < n){
					const float p = __half2float(batchC[row + col * ldc]);
					batchC[row + col * ldc] = __float2half(alpha * cTile[r + c * 64] + beta * p);
				}
			}
		}
	}
}

__global__ __launch_bounds__(256) void GemmExWmmaKernelTN64(const __half* A, const __half* B, __half* C, const int m, const int n, const int k, const int lda, const int ldb, const int ldc, const float alpha, const float beta, const long long strideA, const long long strideB, const long long strideC){
	constexpr int SKEW_A = 8;
	constexpr int SKEW_B = 8;
	constexpr int LDM_A = 16 + SKEW_A;
	constexpr int LDM_B = 16 + SKEW_B;
	const int batch = blockIdx.z;
	const int rowBase = (int)blockIdx.x * 64;
	const int colBase = (int)blockIdx.y * 64;
	if(rowBase >= m || colBase >= n) return;
	const __half* batchA = A + batch * strideA;
	const __half* batchB = B + batch * strideB;
	__half* batchC = C + batch * strideC;
	const int tid = (int)threadIdx.x;
	const int warpId = tid >> 5;
	const int warpRow = warpId & 3;
	const int warpCol = warpId >> 2;
	__shared__ __align__(32) __half aTile[64 * LDM_A];
	__shared__ __align__(32) __half bTile[64 * LDM_B];
	__shared__ __align__(32) float cTile[64 * 64];
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;
	wmma::fill_fragment(acc0, 0.0f);
	wmma::fill_fragment(acc1, 0.0f);
	const bool vecA = (((uintptr_t)batchA & 3u) == 0u) && ((lda & 1) == 0);
	const bool vecB = (((uintptr_t)batchB & 3u) == 0u) && ((ldb & 1) == 0);
	const bool fullA = (rowBase + 63 < m);
	const bool fullB = (colBase + 63 < n);
	for(int kk = 0; kk < k; kk += 16){
		const bool fullK = (kk + 15 < k);
		if(vecA && fullA && fullK){
			for(int idx = tid; idx < 512; idx += 256){
				const int i = idx & 63;
				const int jp = idx >> 6;
				const int j0 = jp * 2;
				const int col = rowBase + i;
				const int row = kk + j0;
				const __half2 v = *reinterpret_cast<const __half2*>(batchA + row + col * lda);
				*reinterpret_cast<__half2*>(aTile + i * LDM_A + j0) = v;
			}
		} else{
			for(int idx = tid; idx < 512; idx += 256){
				const int i = idx & 63;
				const int jp = idx >> 6;
				const int j0 = jp * 2;
				const int col = rowBase + i;
				const int row0 = kk + j0;
				const int row1 = row0 + 1;
				aTile[i * LDM_A + j0] = (col < m && row0 < k) ? batchA[row0 + col * lda] : __float2half(0.0f);
				aTile[i * LDM_A + j0 + 1] = (col < m && row1 < k) ? batchA[row1 + col * lda] : __float2half(0.0f);
			}
		}
		if(vecB && fullB && fullK){
			for(int idx = tid; idx < 512; idx += 256){
				const int rp = idx & 7;
				const int c = idx >> 3;
				const int r0 = rp * 2;
				const int row = kk + r0;
				const int col = colBase + c;
				const __half2 v = *reinterpret_cast<const __half2*>(batchB + row + col * ldb);
				*reinterpret_cast<__half2*>(bTile + r0 + c * LDM_B) = v;
			}
		} else{
			for(int idx = tid; idx < 512; idx += 256){
				const int rp = idx & 7;
				const int c = idx >> 3;
				const int r0 = rp * 2;
				const int row0 = kk + r0;
				const int row1 = row0 + 1;
				const int col = colBase + c;
				bTile[r0 + c * LDM_B] = (row0 < k && col < n) ? batchB[row0 + col * ldb] : __float2half(0.0f);
				bTile[r0 + 1 + c * LDM_B] = (row1 < k && col < n) ? batchB[row1 + col * ldb] : __float2half(0.0f);
			}
		}
		__syncthreads();
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aFrag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag0;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag1;
		const int baseCol = warpCol * 32;
		wmma::load_matrix_sync(aFrag, aTile + (warpRow * 16) * LDM_A, LDM_A);
		wmma::load_matrix_sync(bFrag0, bTile + (baseCol + 0) * LDM_B, LDM_B);
		wmma::load_matrix_sync(bFrag1, bTile + (baseCol + 16) * LDM_B, LDM_B);
		wmma::mma_sync(acc0, aFrag, bFrag0, acc0);
		wmma::mma_sync(acc1, aFrag, bFrag1, acc1);
		__syncthreads();
	}
	const int baseCol = warpCol * 32;
	wmma::store_matrix_sync(cTile + (warpRow * 16) + (baseCol + 0) * 64, acc0, 64, wmma::mem_col_major);
	wmma::store_matrix_sync(cTile + (warpRow * 16) + (baseCol + 16) * 64, acc1, 64, wmma::mem_col_major);
	__syncthreads();
	const bool vecC = (((uintptr_t)batchC & 3u) == 0u) && ((ldc & 1) == 0);
	const bool fullC = (rowBase + 63 < m) && (colBase + 63 < n);
	if(vecC && fullC){
		if(beta == 0.0f){
			for(int idx = tid; idx < 2048; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = colBase + c;
				const float v0 = cTile[r0 + c * 64];
				const float v1 = cTile[r0 + 1 + c * 64];
				*reinterpret_cast<__half2*>(batchC + row + col * ldc) = __floats2half2_rn(alpha * v0, alpha * v1);
			}
		} else{
			for(int idx = tid; idx < 2048; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = colBase + c;
				const float v0 = cTile[r0 + c * 64];
				const float v1 = cTile[r0 + 1 + c * 64];
				const __half2 prevh = *reinterpret_cast<const __half2*>(batchC + row + col * ldc);
				const float p0 = __half2float(__low2half(prevh));
				const float p1 = __half2float(__high2half(prevh));
				*reinterpret_cast<__half2*>(batchC + row + col * ldc) = __floats2half2_rn(alpha * v0 + beta * p0, alpha * v1 + beta * p1);
			}
		}
	} else{
		if(beta == 0.0f){
			for(int idx = tid; idx < 4096; idx += 256){
				const int r = idx & 63;
				const int c = idx >> 6;
				const int row = rowBase + r;
				const int col = colBase + c;
				if(row < m && col < n) batchC[row + col * ldc] = __float2half(alpha * cTile[r + c * 64]);
			}
		} else{
			for(int idx = tid; idx < 4096; idx += 256){
				const int r = idx & 63;
				const int c = idx >> 6;
				const int row = rowBase + r;
				const int col = colBase + c;
				if(row < m && col < n){
					const float p = __half2float(batchC[row + col * ldc]);
					batchC[row + col * ldc] = __float2half(alpha * cTile[r + c * 64] + beta * p);
				}
			}
		}
	}
}

__global__ __launch_bounds__(256) void GemmExWmmaKernelNT64(const __half* A, const __half* B, __half* C, const int m, const int n, const int k, const int lda, const int ldb, const int ldc, const float alpha, const float beta, const long long strideA, const long long strideB, const long long strideC){
	constexpr int SKEW_A = 8;
	constexpr int SKEW_B = 8;
	constexpr int LDM_A = 64 + SKEW_A;
	constexpr int LDM_B = 16 + SKEW_B;
	const int batch = blockIdx.z;
	const int rowBase = (int)blockIdx.x * 64;
	const int colBase = (int)blockIdx.y * 64;
	if(rowBase >= m || colBase >= n) return;
	const __half* batchA = A + batch * strideA;
	const __half* batchB = B + batch * strideB;
	__half* batchC = C + batch * strideC;
	const int tid = (int)threadIdx.x;
	const int warpId = tid >> 5;
	const int warpRow = warpId & 3;
	const int warpCol = warpId >> 2;
	__shared__ __align__(32) __half aTile[64 * LDM_A];
	__shared__ __align__(32) __half bTile[64 * LDM_B];
	__shared__ __align__(32) float cTile[64 * 64];
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;
	wmma::fill_fragment(acc0, 0.0f);
	wmma::fill_fragment(acc1, 0.0f);
	const bool vecA = (((uintptr_t)batchA & 3u) == 0u) && ((lda & 1) == 0);
	const bool vecB = (((uintptr_t)batchB & 3u) == 0u) && ((ldb & 1) == 0);
	const bool fullA = (rowBase + 63 < m);
	const bool fullB = (colBase + 63 < n);
	for(int kk = 0; kk < k; kk += 16){
		const bool fullK = (kk + 15 < k);
		if(vecA && fullA && fullK){
			for(int idx = tid; idx < 512; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = kk + c;
				const __half2 v = *reinterpret_cast<const __half2*>(batchA + row + col * lda);
				*reinterpret_cast<__half2*>(aTile + r0 + c * LDM_A) = v;
			}
		} else{
			for(int idx = tid; idx < 512; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row0 = rowBase + r0;
				const int row1 = row0 + 1;
				const int col = kk + c;
				aTile[r0 + c * LDM_A] = (row0 < m && col < k) ? batchA[row0 + col * lda] : __float2half(0.0f);
				aTile[r0 + 1 + c * LDM_A] = (row1 < m && col < k) ? batchA[row1 + col * lda] : __float2half(0.0f);
			}
		}
		if(vecB && fullB && fullK){
			for(int idx = tid; idx < 512; idx += 256){
				const int jp = idx & 31;
				const int r = idx >> 5;
				const int j0 = jp * 2;
				const int row = colBase + j0;
				const int col = kk + r;
				const __half2 v = *reinterpret_cast<const __half2*>(batchB + row + col * ldb);
				bTile[r + (j0 + 0) * LDM_B] = __low2half(v);
				bTile[r + (j0 + 1) * LDM_B] = __high2half(v);
			}
		} else{
			for(int idx = tid; idx < 1024; idx += 256){
				const int r = idx & 15;
				const int j = idx >> 4;
				const int row = colBase + j;
				const int col = kk + r;
				bTile[r + j * LDM_B] = (row < n && col < k) ? batchB[row + col * ldb] : __float2half(0.0f);
			}
		}
		__syncthreads();
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> aFrag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag0;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag1;
		const int baseCol = warpCol * 32;
		wmma::load_matrix_sync(aFrag, aTile + warpRow * 16, LDM_A);
		wmma::load_matrix_sync(bFrag0, bTile + (baseCol + 0) * LDM_B, LDM_B);
		wmma::load_matrix_sync(bFrag1, bTile + (baseCol + 16) * LDM_B, LDM_B);
		wmma::mma_sync(acc0, aFrag, bFrag0, acc0);
		wmma::mma_sync(acc1, aFrag, bFrag1, acc1);
		__syncthreads();
	}
	const int baseCol = warpCol * 32;
	wmma::store_matrix_sync(cTile + (warpRow * 16) + (baseCol + 0) * 64, acc0, 64, wmma::mem_col_major);
	wmma::store_matrix_sync(cTile + (warpRow * 16) + (baseCol + 16) * 64, acc1, 64, wmma::mem_col_major);
	__syncthreads();
	const bool vecC = (((uintptr_t)batchC & 3u) == 0u) && ((ldc & 1) == 0);
	const bool fullC = (rowBase + 63 < m) && (colBase + 63 < n);
	if(vecC && fullC){
		if(beta == 0.0f){
			for(int idx = tid; idx < 2048; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = colBase + c;
				const float v0 = cTile[r0 + c * 64];
				const float v1 = cTile[r0 + 1 + c * 64];
				*reinterpret_cast<__half2*>(batchC + row + col * ldc) = __floats2half2_rn(alpha * v0, alpha * v1);
			}
		} else{
			for(int idx = tid; idx < 2048; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = colBase + c;
				const float v0 = cTile[r0 + c * 64];
				const float v1 = cTile[r0 + 1 + c * 64];
				const __half2 prevh = *reinterpret_cast<const __half2*>(batchC + row + col * ldc);
				const float p0 = __half2float(__low2half(prevh));
				const float p1 = __half2float(__high2half(prevh));
				*reinterpret_cast<__half2*>(batchC + row + col * ldc) = __floats2half2_rn(alpha * v0 + beta * p0, alpha * v1 + beta * p1);
			}
		}
	} else{
		if(beta == 0.0f){
			for(int idx = tid; idx < 4096; idx += 256){
				const int r = idx & 63;
				const int c = idx >> 6;
				const int row = rowBase + r;
				const int col = colBase + c;
				if(row < m && col < n) batchC[row + col * ldc] = __float2half(alpha * cTile[r + c * 64]);
			}
		} else{
			for(int idx = tid; idx < 4096; idx += 256){
				const int r = idx & 63;
				const int c = idx >> 6;
				const int row = rowBase + r;
				const int col = colBase + c;
				if(row < m && col < n){
					const float p = __half2float(batchC[row + col * ldc]);
					batchC[row + col * ldc] = __float2half(alpha * cTile[r + c * 64] + beta * p);
				}
			}
		}
	}
}

__global__ __launch_bounds__(256) void GemmExWmmaKernelTT64(const __half* A, const __half* B, __half* C, const int m, const int n, const int k, const int lda, const int ldb, const int ldc, const float alpha, const float beta, const long long strideA, const long long strideB, const long long strideC){
	constexpr int SKEW_A = 8;
	constexpr int SKEW_B = 8;
	constexpr int LDM_A = 16 + SKEW_A;
	constexpr int LDM_B = 16 + SKEW_B;
	const int batch = blockIdx.z;
	const int rowBase = (int)blockIdx.x * 64;
	const int colBase = (int)blockIdx.y * 64;
	if(rowBase >= m || colBase >= n) return;
	const __half* batchA = A + batch * strideA;
	const __half* batchB = B + batch * strideB;
	__half* batchC = C + batch * strideC;
	const int tid = (int)threadIdx.x;
	const int warpId = tid >> 5;
	const int warpRow = warpId & 3;
	const int warpCol = warpId >> 2;
	__shared__ __align__(32) __half aTile[64 * LDM_A];
	__shared__ __align__(32) __half bTile[64 * LDM_B];
	__shared__ __align__(32) float cTile[64 * 64];
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc0;
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc1;
	wmma::fill_fragment(acc0, 0.0f);
	wmma::fill_fragment(acc1, 0.0f);
	const bool vecA = (((uintptr_t)batchA & 3u) == 0u) && ((lda & 1) == 0);
	const bool vecB = (((uintptr_t)batchB & 3u) == 0u) && ((ldb & 1) == 0);
	const bool fullA = (rowBase + 63 < m);
	const bool fullB = (colBase + 63 < n);
	for(int kk = 0; kk < k; kk += 16){
		const bool fullK = (kk + 15 < k);
		if(vecA && fullA && fullK){
			for(int idx = tid; idx < 512; idx += 256){
				const int i = idx & 63;
				const int jp = idx >> 6;
				const int j0 = jp * 2;
				const int col = rowBase + i;
				const int row = kk + j0;
				const __half2 v = *reinterpret_cast<const __half2*>(batchA + row + col * lda);
				*reinterpret_cast<__half2*>(aTile + i * LDM_A + j0) = v;
			}
		} else{
			for(int idx = tid; idx < 512; idx += 256){
				const int i = idx & 63;
				const int jp = idx >> 6;
				const int j0 = jp * 2;
				const int col = rowBase + i;
				const int row0 = kk + j0;
				const int row1 = row0 + 1;
				aTile[i * LDM_A + j0] = (col < m && row0 < k) ? batchA[row0 + col * lda] : __float2half(0.0f);
				aTile[i * LDM_A + j0 + 1] = (col < m && row1 < k) ? batchA[row1 + col * lda] : __float2half(0.0f);
			}
		}
		if(vecB && fullB && fullK){
			for(int idx = tid; idx < 512; idx += 256){
				const int jp = idx & 31;
				const int r = idx >> 5;
				const int j0 = jp * 2;
				const int row = colBase + j0;
				const int col = kk + r;
				const __half2 v = *reinterpret_cast<const __half2*>(batchB + row + col * ldb);
				bTile[r + (j0 + 0) * LDM_B] = __low2half(v);
				bTile[r + (j0 + 1) * LDM_B] = __high2half(v);
			}
		} else{
			for(int idx = tid; idx < 1024; idx += 256){
				const int r = idx & 15;
				const int j = idx >> 4;
				const int row = colBase + j;
				const int col = kk + r;
				bTile[r + j * LDM_B] = (row < n && col < k) ? batchB[row + col * ldb] : __float2half(0.0f);
			}
		}
		__syncthreads();
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> aFrag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag0;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag1;
		const int baseCol = warpCol * 32;
		wmma::load_matrix_sync(aFrag, aTile + (warpRow * 16) * LDM_A, LDM_A);
		wmma::load_matrix_sync(bFrag0, bTile + (baseCol + 0) * LDM_B, LDM_B);
		wmma::load_matrix_sync(bFrag1, bTile + (baseCol + 16) * LDM_B, LDM_B);
		wmma::mma_sync(acc0, aFrag, bFrag0, acc0);
		wmma::mma_sync(acc1, aFrag, bFrag1, acc1);
		__syncthreads();
	}
	const int baseCol = warpCol * 32;
	wmma::store_matrix_sync(cTile + (warpRow * 16) + (baseCol + 0) * 64, acc0, 64, wmma::mem_col_major);
	wmma::store_matrix_sync(cTile + (warpRow * 16) + (baseCol + 16) * 64, acc1, 64, wmma::mem_col_major);
	__syncthreads();
	const bool vecC = (((uintptr_t)batchC & 3u) == 0u) && ((ldc & 1) == 0);
	const bool fullC = (rowBase + 63 < m) && (colBase + 63 < n);
	if(vecC && fullC){
		if(beta == 0.0f){
			for(int idx = tid; idx < 2048; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = colBase + c;
				const float v0 = cTile[r0 + c * 64];
				const float v1 = cTile[r0 + 1 + c * 64];
				*reinterpret_cast<__half2*>(batchC + row + col * ldc) = __floats2half2_rn(alpha * v0, alpha * v1);
			}
		} else{
			for(int idx = tid; idx < 2048; idx += 256){
				const int rp = idx & 31;
				const int c = idx >> 5;
				const int r0 = rp * 2;
				const int row = rowBase + r0;
				const int col = colBase + c;
				const float v0 = cTile[r0 + c * 64];
				const float v1 = cTile[r0 + 1 + c * 64];
				const __half2 prevh = *reinterpret_cast<const __half2*>(batchC + row + col * ldc);
				const float p0 = __half2float(__low2half(prevh));
				const float p1 = __half2float(__high2half(prevh));
				*reinterpret_cast<__half2*>(batchC + row + col * ldc) = __floats2half2_rn(alpha * v0 + beta * p0, alpha * v1 + beta * p1);
			}
		}
	} else{
		if(beta == 0.0f){
			for(int idx = tid; idx < 4096; idx += 256){
				const int r = idx & 63;
				const int c = idx >> 6;
				const int row = rowBase + r;
				const int col = colBase + c;
				if(row < m && col < n) batchC[row + col * ldc] = __float2half(alpha * cTile[r + c * 64]);
			}
		} else{
			for(int idx = tid; idx < 4096; idx += 256){
				const int r = idx & 63;
				const int c = idx >> 6;
				const int row = rowBase + r;
				const int col = colBase + c;
				if(row < m && col < n){
					const float p = __half2float(batchC[row + col * ldc]);
					batchC[row + col * ldc] = __float2half(alpha * cTile[r + c * 64] + beta * p);
				}
			}
		}
	}
}

bool CanUseWmma(const cudaDataType Atype, const cudaDataType Btype, const cudaDataType Ctype, const cudaDataType computeType, const CLNNOpT transa, const CLNNOpT transb){
	if(Atype != CUDA_R_16F || Btype != CUDA_R_16F || Ctype != CUDA_R_16F) return false;
	if(computeType != CUDA_R_32F) return false;
	if(transa != CLNN_OP_N && transa != CLNN_OP_T || transb != CLNN_OP_N && transb != CLNN_OP_T) return false;
	cudaDeviceProp prop{};
	if(cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
	return prop.major == 7;
}

float ReadAlpha(const void* alpha){ return alpha ? *static_cast<const float*>(alpha) : 1.0f; }
float ReadBeta(const void* beta){ return beta ? *static_cast<const float*>(beta) : 0.0f; }

CLNNStatusT CLNNGemmEx(const CLNNOpT transa, const CLNNOpT transb, const int m, const int n, const int k, const void* alpha, const void* A, const cudaDataType Atype, const int lda, const void* B, const cudaDataType Btype, const int ldb, const void* beta, void* C, const cudaDataType Ctype, const int ldc, const cudaDataType computeType){
	if(!CanUseWmma(Atype, Btype, Ctype, computeType, transa, transb)) return CLNN_STATUS_NOT_SUPPORTED;
	cudaStream_t stream = nullptr;
	const float a = ReadAlpha(alpha);
	const float b = ReadBeta(beta);
	const dim3 block(256, 1, 1);
	const dim3 grid(DivCeil(m, 64), DivCeil(n, 64), 1);
	if(transa == CLNN_OP_N && transb == CLNN_OP_N) GemmExWmmaKernelNN64<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, a, b, 0, 0, 0);
	else if(transa == CLNN_OP_T && transb == CLNN_OP_N) GemmExWmmaKernelTN64<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, a, b, 0, 0, 0);
	else if(transa == CLNN_OP_N && transb == CLNN_OP_T) GemmExWmmaKernelNT64<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, a, b, 0, 0, 0);
	else GemmExWmmaKernelTT64<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, a, b, 0, 0, 0);
	if(cudaPeekAtLastError() != cudaSuccess) return CLNN_STATUS_EXECUTION_FAILED;
	return CLNN_STATUS_SUCCESS;
}

CLNNStatusT CLNNGemmStridedBatchedEx(const CLNNOpT transa, const CLNNOpT transb, const int m, const int n, const int k, const void* alpha, const void* A, const cudaDataType Atype, const int lda, const long long int strideA, const void* B, const cudaDataType Btype, const int ldb, const long long int strideB, const void* beta, void* C, const cudaDataType Ctype, const int ldc, const long long int strideC, const int batchCount, const cudaDataType computeType){
	if(batchCount <= 0) return CLNN_STATUS_INVALID_VALUE;
	if(!CanUseWmma(Atype, Btype, Ctype, computeType, transa, transb)) return CLNN_STATUS_NOT_SUPPORTED;
	cudaStream_t stream = nullptr;
	const float a = ReadAlpha(alpha);
	const float b = ReadBeta(beta);
	const dim3 block(256, 1, 1);
	const dim3 grid(DivCeil(m, 64), DivCeil(n, 64), batchCount);
	if(transa == CLNN_OP_N && transb == CLNN_OP_N) GemmExWmmaKernelNN64<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, a, b, strideA, strideB, strideC);
	else if(transa == CLNN_OP_T && transb == CLNN_OP_N) GemmExWmmaKernelTN64<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, a, b, strideA, strideB, strideC);
	else if(transa == CLNN_OP_N && transb == CLNN_OP_T) GemmExWmmaKernelNT64<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, a, b, strideA, strideB, strideC);
	else GemmExWmmaKernelTT64<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, a, b, strideA, strideB, strideC);
	if(cudaPeekAtLastError() != cudaSuccess) return CLNN_STATUS_EXECUTION_FAILED;
	return CLNN_STATUS_SUCCESS;
}