#pragma once
#ifdef CLNN_SHARED
#    ifdef CLNN_BUILD
#        define CLNN_API __declspec(dllexport)
#    else
#        define CLNN_API __declspec(dllimport)
#    endif
#else
#    define CLNN_API
#endif
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <library_types.h>
#include <stdexcept>
#include <iostream>
#include <string>
enum WeightInitMethod{
	He, Xavier
};
typedef enum{
	CLNN_OP_N = 0,
	CLNN_OP_T = 1,
	CLNN_OP_C = 2,
} CLNNOpT;
typedef enum{
	CLNN_STATUS_SUCCESS = 0,
	CLNN_STATUS_NOT_INITIALIZED = 1,
	CLNN_STATUS_ALLOC_FAILED = 3,
	CLNN_STATUS_INVALID_VALUE = 7,
	CLNN_STATUS_ARCH_MISMATCH = 8,
	CLNN_STATUS_MAPPING_ERROR = 11,
	CLNN_STATUS_EXECUTION_FAILED = 13,
	CLNN_STATUS_INTERNAL_ERROR = 14,
	CLNN_STATUS_NOT_SUPPORTED = 15
} CLNNStatusT;
struct PixARGB{
	unsigned char B;
	unsigned char G;
	unsigned char R;
	unsigned char A;
};
struct PixRGB{
	unsigned char B;
	unsigned char G;
	unsigned char R;
};
struct AdamWHalfTask{
	__half* params;
	const __half* grads;
	__half* m;
	__half* v;
	int size;
	float weightDecay;
};
struct AdamWFloatTask{
	float* params;
	const float* grads;
	float* m;
	float* v;
	int size;
	float weightDecay;
};
CLNN_API const char* clnnGetErrorString(CLNNStatusT status);
#define checkCLNN(status) { \
	const auto err = status; \
    if (err != CLNN_STATUS_SUCCESS) { \
        std::cerr << "\nError: " << clnnGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("Error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + clnnGetErrorString(err)); \
    } \
}
#define checkCUDNN(status) { \
	const auto err = status; \
    if (err != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "\ncuDNN error: " << cudnnGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuDNN error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudnnGetErrorString(err)); \
    } \
}
#define checkCUDA(status) { \
	const auto err = status; \
    if (err != cudaSuccess) { \
        std::cerr << "\nCUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(err)); \
    } \
}
template<class Ta, class Tb>
Ta DivCeil(Ta a, Tb b){ return (a + b - 1)/b; }
template<class T>
void CUDAMallocZero(T** ptr, size_t size){
	checkCUDA(cudaMalloc(reinterpret_cast<void**>(ptr), size));
	checkCUDA(cudaMemset(*ptr, 0, size));
}
CLNN_API void InitCUDA();
CLNN_API CLNNStatusT InitCublas();
CLNN_API void LossStats(const __half* dPredictions, const float* dTargets, int numButs, int numCtrls, int batchSize, float* butLoss, float* axesLoss);
CLNN_API void LossBackprop(__half* dGradient, const __half* dPredictions, const float* dTargets, float clip, int size, int numCtrls, int numButs, int batchSize);
CLNN_API void BlockShiftHalf(__half* dPtr, int shiftBy, int blocksToShift);
CLNN_API void ConvertByteToHalf(const unsigned char* input, __half* output, size_t size, bool normalize);
CLNN_API void ConvertHalfToByte(const __half* input, unsigned char* output, size_t size, bool normalize);
CLNN_API void ConvertFloatToHalf(const float* input, __half* output, size_t size);
CLNN_API void ConvertHalfToFloat(const __half* input, float* output, size_t size);
CLNN_API void ConvertFloatToHalfScale(__half* halfWeights, const float* weights, size_t size, float scale);
CLNN_API void ARGBtoRGB(unsigned char* src, unsigned char* dst, size_t n);
CLNN_API void ARGBtoRGBplanar(const unsigned char* src, unsigned char* dst, size_t n);
CLNN_API void SGDHalf(__half* params, const __half* grads, int size, float learningRate, float weightDecay);
CLNN_API void SGDFloat(float* params, const float* grads, int size, float learningRate, float weightDecay);
CLNN_API void AdamWHalf(__half* params, const __half* grads, __half* m, __half* v, float lr, int t, float weightDecay, int size);
CLNN_API void AdamWFloat(float* params, const float* grads, float* m, float* v, float learningRate, int t, float weightDecay, int size);
CLNN_API void AdamWHalfMulti(const AdamWHalfTask* tasks, int taskCount, int totalSize, float lr, int t);
CLNN_API void AdamWFloatMulti(const AdamWFloatTask* tasks, int taskCount, int totalSize, float lr, int t);
CLNN_API bool IsnanHalf(const __half* data, int size);
CLNNStatusT CLNNGemmEx(CLNNOpT transa, CLNNOpT transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc, cudaDataType computeType);
CLNNStatusT CLNNGemmStridedBatchedEx(CLNNOpT transa, CLNNOpT transb, int m, int n, int k, const void* alpha, const void* A, cudaDataType Atype, int lda, long long int strideA, const void* B, cudaDataType Btype, int ldb, long long int strideB, const void* beta, void* C, cudaDataType Ctype, int ldc, long long int strideC, int batchCount, cudaDataType computeType);