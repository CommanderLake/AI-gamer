#pragma once
#include <algorithm>
#include <cstdint>
#include <map>
#include <cublas_v2.h>
#include <iostream>
const char* cublasGetErrorString(cublasStatus_t status);
#define checkCUBLAS(status) { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << cublasGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        throw std::exception(); \
    } \
}
#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        throw std::exception(); \
    } \
}
#define checkCUDA(status) { \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at line " << __LINE__ << std::endl; \
        throw std::exception(); \
    } \
}
struct __half;
struct InputRecord{
	uint16_t keyStates; // Bitmask for key states
	int32_t mouseDeltaX; // Total mouse delta X movement
	int32_t mouseDeltaY; // Total mouse delta Y movement
	unsigned char* state_data = nullptr;
	~InputRecord(){
		if(state_data){
			_mm_free(state_data);
		}
	}
};
void ClearScreen(char fill = ' ');
const std::string fileName("E:\\training_data.bin");
extern std::map<int, int> keyMap;
int _ConvertSMVer2Cores(int major, int minor);
void H2F128Asm(float* dst, __half* src, int numElements);
void PrintDataHalf(const __half* data, size_t size, const char* label);
void PrintDataFloat(const float* data, size_t size, const char* label);
void PrintDataFloatHost(const float* data, size_t size, const char* label);
void PrintDataCharHost(const unsigned char* data, size_t size, const char* label);
void CheckData(const __half* data, size_t size, const char* label);
extern "C" float MseLoss(const __half* d_predictions, const float* d_targets, int size);
extern "C" void ConvertAndNormalize(unsigned char* input, __half* output, size_t size);
extern "C" void ConvertFloatToHalf(float* src, __half* dst, size_t n);
extern "C" void ConvertHalfToFloat(__half* src, float* dst, size_t n);
extern "C" void HeInit(__half* weightHalf, int numWeights, float fanIn);
extern "C" void GemmHff(int M, int N, int K, bool transA, bool transB, const half *A, int lda, const float *B, int ldb, float *C, int ldc);
extern "C" void SGDHalf(__half* param, float learningRate, const __half* gradParam, int size);
extern "C" void SGDFloat(float* param, float learningRate, const float* gradParam, int size);
extern "C" void AdamHalf(__half* param, float* m, float* v, float learningRate, const __half* gradParam, int size, int t);
extern "C" void AdamFloat(float* param, float* m, float* v, float learningRate, const float* gradParam, int size, int t);
extern "C" void AdamWHalf(__half* param, float* m, float* v, float learningRate, const __half* gradParam, int size, int t, float weightDecay);
extern "C" void AdamWFloat(float* param, float* m, float* v, float learningRate, const float* gradParam, int size, int t, float weightDecay);
extern "C" void Gradient(__half* d_gradient, const __half* d_predictions, const float* d_targets, int batchSize, int n, float scale);
extern "C" void BiasGradient(const __half* gradInput, __half* gradBias, int c, int batchSize);
extern "C" void ClipGrads(__half* grad, int size);
extern "C" void Scale(__half* data, int size, float scale);
extern "C" void LeakyRelu(__half* data, int size, float negativeSlope);
extern "C" void LeakyReluBackward(__half* gradient, const __half* inData, int size, float negativeSlope);
extern "C" void SigmoidForward(__half* data, int numSigmoidOutputs, int batchSize, int outputSize);
extern "C" void SigmoidBackward(__half* grad, const __half* data, int numSigmoidOutputs, int batchSize, int outputSize);
extern "C" void LayerNormForward(__half* output, const __half* data, const float* gamma, const float* beta, float* mean, float* variance, int N, int C, int HW, float epsilon);
extern "C" void LayerNormBackward(__half* gradIn, const __half* gradOut, const __half* data, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW, float epsilon);
extern "C" void SimpleLayerNormForward(__half* output, const __half* data, const float* gamma, const float* beta, float* mean, float* variance, int N, int C, int HW, float epsilon);