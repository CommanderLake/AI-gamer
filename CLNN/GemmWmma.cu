#define __CUDACC__
#include "CuCommon.cuh"
#include <mma.h>
using namespace nvcuda;
__device__ __forceinline__ __half LoadHalfElement(const __half* ptr, const int row, const int col, const int ld){ return ptr[row + col * ld]; }
__device__ __forceinline__ __half GetOpA(const __half* A, const cublasOperation_t transa, const int row, const int col, const int lda){ return transa == CUBLAS_OP_N ? LoadHalfElement(A, row, col, lda) : LoadHalfElement(A, col, row, lda); }
__device__ __forceinline__ __half GetOpB(const __half* B, const cublasOperation_t transb, const int row, const int col, const int ldb){ return transb == CUBLAS_OP_N ? LoadHalfElement(B, row, col, ldb) : LoadHalfElement(B, col, row, ldb); }
__global__ void GemmExWmmaKernel(const __half* A, const __half* B, __half* C, const int m, const int n, const int k, const int lda, const int ldb, const int ldc, const float alpha, const float beta, const cublasOperation_t transa, const cublasOperation_t transb, const long long strideA, const long long strideB, const long long strideC){
	const int batch = blockIdx.z;
	const int tileRow = blockIdx.x;
	const int tileCol = blockIdx.y;
	const int rowBase = tileRow * 16;
	const int colBase = tileCol * 16;
	if(rowBase >= m || colBase >= n) return;
	const __half* batchA = A + batch * strideA;
	const __half* batchB = B + batch * strideB;
	__half* batchC = C + batch * strideC;
	__shared__ __half aTile[16 * 16];
	__shared__ __half bTile[16 * 16];
	wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
	fill_fragment(acc, 0.0f);
	for(int kk = 0; kk < k; kk += 16){
		for(int idx = threadIdx.x; idx < 256; idx += blockDim.x){
			const int r = idx & 15;
			const int c = idx >> 4;
			const int aRow = rowBase + r;
			const int aCol = kk + c;
			const int bRow = kk + r;
			const int bCol = colBase + c;
			aTile[idx] = aRow < m && aCol < k ? GetOpA(batchA, transa, aRow, aCol, lda) : __float2half(0.0f);
			bTile[idx] = bRow < k && bCol < n ? GetOpB(batchB, transb, bRow, bCol, ldb) : __float2half(0.0f);
		}
		__syncthreads();
		wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> aFrag;
		wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bFrag;
		load_matrix_sync(aFrag, aTile, 16);
		load_matrix_sync(bFrag, bTile, 16);
		mma_sync(acc, aFrag, bFrag, acc);
		__syncthreads();
	}
	for(int idx = threadIdx.x; idx < 256; idx += blockDim.x){
		const int r = idx & 15;
		const int c = idx >> 4;
		const int row = rowBase + r;
		const int col = colBase + c;
		if(row < m && col < n){
			const int fragIdx = c * 16 + r;
			const float prev = __half2float(batchC[row + col * ldc]);
			batchC[row + col * ldc] = __float2half(alpha * acc.x[fragIdx] + beta * prev);
		}
	}
}
bool CanUseWmma(const cudaDataType Atype, const cudaDataType Btype, const cudaDataType Ctype, const cudaDataType computeType, const cublasOperation_t transa, const cublasOperation_t transb){
	if(Atype != CUDA_R_16F || Btype != CUDA_R_16F || Ctype != CUDA_R_16F) return false;
	if(computeType != CUDA_R_32F) return false;
	if(transa != CUBLAS_OP_N && transa != CUBLAS_OP_T || transb != CUBLAS_OP_N && transb != CUBLAS_OP_T) return false;
	cudaDeviceProp prop{};
	if(cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
	return prop.major == 7;
}
float ReadScale(const void* scale){ return scale ? *static_cast<const float*>(scale) : 0.0f; }
cublasStatus_t CLNNGemmEx(const cublasHandle_t handle, const cublasOperation_t transa, const cublasOperation_t transb, const int m, const int n, const int k, const void* alpha, const void* A, const cudaDataType Atype, const int lda, const void* B, const cudaDataType Btype, const int ldb, const void* beta, void* C, const cudaDataType Ctype, const int ldc, const cudaDataType computeType, const cublasGemmAlgo_t algo){
	if(!CanUseWmma(Atype, Btype, Ctype, computeType, transa, transb)){ return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo); }
	cudaStream_t stream = nullptr;
	if(cublasGetStream(handle, &stream) != CUBLAS_STATUS_SUCCESS) return CUBLAS_STATUS_INTERNAL_ERROR;
	const dim3 block(32, 1, 1);
	const dim3 grid(DivCeil(m, 16), DivCeil(n, 16), 1);
	GemmExWmmaKernel<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, ReadScale(alpha), ReadScale(beta), transa, transb, 0, 0, 0);
	if(cudaPeekAtLastError() != cudaSuccess) return CUBLAS_STATUS_EXECUTION_FAILED;
	return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t CLNNGemmStridedBatchedEx(const cublasHandle_t handle, const cublasOperation_t transa, const cublasOperation_t transb, const int m, const int n, const int k, const void* alpha, const void* A, const cudaDataType Atype, const int lda, const long long int strideA, const void* B, const cudaDataType Btype, const int ldb, const long long int strideB, const void* beta, void* C, const cudaDataType Ctype, const int ldc, const long long int strideC, const int batchCount, const cudaDataType computeType, const cublasGemmAlgo_t algo){
	if(!CanUseWmma(Atype, Btype, Ctype, computeType, transa, transb) || batchCount <= 0){ return cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo); }
	cudaStream_t stream = nullptr;
	if(cublasGetStream(handle, &stream) != CUBLAS_STATUS_SUCCESS) return CUBLAS_STATUS_INTERNAL_ERROR;
	const dim3 block(32, 1, 1);
	const dim3 grid(DivCeil(m, 16), DivCeil(n, 16), batchCount);
	GemmExWmmaKernel<<<grid, block, 0, stream>>>(static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), m, n, k, lda, ldb, ldc, ReadScale(alpha), ReadScale(beta), transa, transb, strideA, strideB, strideC);
	if(cudaPeekAtLastError() != cudaSuccess) return CUBLAS_STATUS_EXECUTION_FAILED;
	return CUBLAS_STATUS_SUCCESS;
}