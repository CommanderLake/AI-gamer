#include "CuCommon.cuh"
#include <cublas_v2.h>
static constexpr cublasOperation_t ToCublasOp(const CLNNOpT op){
	if(op == CLNN_OP_T || op == CLNN_OP_C) return CUBLAS_OP_T;
	return CUBLAS_OP_N;
}
static constexpr CLNNStatusT ToClnnStatus(const cublasStatus_t status){
	switch(status){
		case CUBLAS_STATUS_SUCCESS: return CLNN_STATUS_SUCCESS;
		case CUBLAS_STATUS_NOT_INITIALIZED: return CLNN_STATUS_NOT_INITIALIZED;
		case CUBLAS_STATUS_ALLOC_FAILED: return CLNN_STATUS_ALLOC_FAILED;
		case CUBLAS_STATUS_INVALID_VALUE: return CLNN_STATUS_INVALID_VALUE;
		case CUBLAS_STATUS_ARCH_MISMATCH: return CLNN_STATUS_ARCH_MISMATCH;
		case CUBLAS_STATUS_MAPPING_ERROR: return CLNN_STATUS_MAPPING_ERROR;
		case CUBLAS_STATUS_EXECUTION_FAILED: return CLNN_STATUS_EXECUTION_FAILED;
		case CUBLAS_STATUS_INTERNAL_ERROR: return CLNN_STATUS_INTERNAL_ERROR;
		case CUBLAS_STATUS_NOT_SUPPORTED: return CLNN_STATUS_NOT_SUPPORTED;
		default: return CLNN_STATUS_INTERNAL_ERROR;
	}
}
cublasHandle_t handle = nullptr;
CLNNStatusT InitCublas(){
	cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
	if(handle == nullptr){
		status = cublasCreate(&handle);
		if(status == CUBLAS_STATUS_SUCCESS) status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
	}
	return ToClnnStatus(status);
}
CLNNStatusT CLNNGemmEx(const CLNNOpT transa, const CLNNOpT transb, const int m, const int n, const int k, const void* alpha, const void* A, const cudaDataType Atype, const int lda, const void* B, const cudaDataType Btype, const int ldb, const void* beta, void* C, const cudaDataType Ctype, const int ldc, const cudaDataType computeType){
	const auto status = cublasGemmEx(handle, ToCublasOp(transa), ToCublasOp(transb), m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	return ToClnnStatus(status);
}
CLNNStatusT CLNNGemmStridedBatchedEx(const CLNNOpT transa, const CLNNOpT transb, const int m, const int n, const int k, const void* alpha, const void* A, const cudaDataType Atype, const int lda, const long long sA, const void* B, const cudaDataType Btype, const int ldb, const long long sB, const void* beta, void* C, const cudaDataType Ctype, const int ldc, const long long sC, const int batchCount, const cudaDataType computeType){
	if(batchCount <= 0) return CLNN_STATUS_INVALID_VALUE;
	const auto status = cublasGemmStridedBatchedEx(handle, ToCublasOp(transa), ToCublasOp(transb), m, n, k, alpha, A, Atype, lda, sA, B, Btype, ldb, sB, beta, C, Ctype, ldc, sC, batchCount, computeType, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	return ToClnnStatus(status);
}