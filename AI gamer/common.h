#pragma once
#include <algorithm>
#include <vector>
#include <cstdint>
#include <map>
#include <cuda_runtime_api.h>
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
void clear_screen(char fill = ' ');
const std::string fileName("E:\\training_data.bin");
extern std::map<int, int> keyMap;
int _ConvertSMVer2Cores(int major, int minor);
void h2f128asm(float* dst, __half* src, int num_elements);
void printDataHalf(const __half* data, size_t size, const char* label);
void printDataFloat(const float* data, size_t size, const char* label);
void printDataFloatHost(const float* data, size_t size, const char* label);
void checkData(const __half* data, size_t size, const char* label);
extern "C" float mseLoss(const __half* d_predictions, const float* d_targets, int size);
extern "C" void convertAndNormalize(unsigned char* input, __half* output, size_t size);
extern "C" void convertFloatToHalf(float* src, __half* dst, size_t n);
extern "C" void convertHalfToFloat(__half* src, float* dst, size_t n);
extern "C" void HeInit(__half* weightHalf, int numWeights, float fanIn);
extern "C" void Hgemv(const half *A, const half *B, half *C, int N, int K);
extern "C" void SGDHalf(__half* param, float learningRate, const __half* gradParam, int size);
extern "C" void SGDFloat(float* param, float learningRate, const float* gradParam, int size);
extern "C" void AdamHalf(__half* param, float* m, float* v, float learningRate, const __half* gradParam, int size, int t);
extern "C" void AdamFloat(float* param, float* m, float* v, float learningRate, const float* gradParam, int size, int t);
extern "C" void AdamWHalf(__half* param, float* m, float* v, float learningRate, const __half* gradParam, int size, int t, float weightDecay);
extern "C" void AdamWFloat(float* param, float* m, float* v, float learningRate, const float* gradParam, int size, int t, float weightDecay);
extern "C" void gradient(__half* d_gradient, const __half* d_predictions, const float* d_targets, int batchSize, int outputSize);
extern "C" void biasGradient(const __half* gradInput, __half* gradBias, int c, int batchSize);
extern "C" void clipGrads(__half* grad, int size);
extern "C" void scale(__half* data, int size, float scale);
extern "C" void leakyRelu(__half* data, int size, float negativeSlope);
extern "C" void leakyReluBackward(const __half* gradIn, const __half* inData, __half* gradOut, int size, float negativeSlope);