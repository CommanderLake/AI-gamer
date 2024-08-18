#pragma once
#include "ThreadPool.h"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <unordered_map>
#define WM_USER_STOP_CAPTURE (WM_USER + 1)
#define WM_USER_START_CAPTURE (WM_USER + 2)
#define WM_USER_CAPTURE_FRAME (WM_USER + 3)
const char* cublasGetErrorString(cublasStatus_t status);
#define checkCUBLAS(status) { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << cublasGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuBLAS error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cublasGetErrorString(status)); \
    } \
}

#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuDNN error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudnnGetErrorString(status)); \
    } \
}

#define checkCUDA(status) { \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(status)); \
    } \
}
struct __half;
struct InputRecord{
	unsigned short keyStates;
	int mouseDeltaX;
	int mouseDeltaY;
	unsigned char* stateData = nullptr;
	~InputRecord(){
		if(stateData){
			_mm_free(stateData);
		}
	}
};
struct StateBatch{
	int batchSize;
	int stateSize;
	unsigned short* keyStates;
	int* mouseDeltaX;
	int* mouseDeltaY;
	unsigned char* stateData = nullptr;
	explicit StateBatch(const int batchSize, int stateSize): batchSize(batchSize), stateSize(stateSize){
		keyStates = static_cast<unsigned short*>(malloc(batchSize * sizeof(unsigned short)));
		mouseDeltaX = static_cast<int*>(malloc(batchSize * sizeof(int)));
		mouseDeltaY = static_cast<int*>(malloc(batchSize * sizeof(int)));
		if(!keyStates || !mouseDeltaX || !mouseDeltaY){ throw std::bad_alloc(); }
		//stateData = static_cast<unsigned char*>(_mm_malloc(stateSize*batchSize, 64));
		checkCUDA(cudaMallocHost(reinterpret_cast<void**>(&stateData), stateSize*batchSize));
	}
	~StateBatch(){
		free(keyStates);
		free(mouseDeltaX);
		free(mouseDeltaY);
		//_mm_free(stateData);
		cudaFreeHost(stateData);
	}
};
extern std::vector<std::string> trainDataInFiles;
extern std::unordered_map<std::string, std::vector<std::streampos>> fileRecordIndex;
extern std::size_t stateSize;
extern ThreadPool threadPool;
void LoadBatch(StateBatch* batch, int seqLength, int numSequences);
const std::string trainDataOutFileName("E:\\training_data.bin");
const std::string ckptFileName("E:\\AIGamer.ckpt");
const std::string optFileName("E:\\AIGamer.opt");
const std::string ckptFileNameDisc("E:\\AIGamerDisc.ckpt");
const std::string optFileNameDisc("E:\\AIGamerDisc.opt");
constexpr int numButs_ = 14;
constexpr int numAxes_ = 2;
constexpr int numCtrls_ = numButs_ + numAxes_;
extern unsigned char keyMap[14];
extern unsigned char keyReMap[14];
void ClearScreen(char fill = ' ');
int ConvertSmVer2Cores(int major, int minor);
void H2F128Asm(float* dst, __half* src, int numElements);
void PrintDataHalf(const __half* data, size_t size, const char* label);
void PrintDataHalf2(const __half* data, size_t size, const char* label);
void PrintDataFloat(const float* data, size_t size, const char* label);
void PrintDataFloatHost(const float* data, size_t size, const char* label);
void PrintDataCharHost(const unsigned char* data, size_t size, const char* label);
extern "C" void InitCUDA();
extern "C" float MseLoss(const __half* dPredictions, const float* dTargets, int size);
extern "C" void ConvertAndNormalize(__half* output, unsigned char* input, size_t size);
extern "C" void UnConvertAndUnNormalize(unsigned char* output, const __half* input, size_t size);
extern "C" void ConvertFloatToHalf(float* src, __half* dst, size_t n);
extern "C" void ConvertHalfToFloat(__half* src, float* dst, size_t n);
extern "C" void HeInit(__half* weightHalf, int numWeights, float fanIn);
extern "C" void SGDHalf(__half* param, float learningRate, const __half* gradParam, int size);
extern "C" void SGDFloat(float* param, float learningRate, const float* gradParam, int size);
extern "C" void AdamWHalf(__half* params, const __half* grads, float* m, float* v, float lr, int t, float weightDecay, int size);
extern "C" void AdamWFloat(float* params, const float* grads, float* m, float* v, float learningRate, int t, float weightDecay, int size);
extern "C" void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, int size, float scale, float clip);
extern "C" void BiasGradient(const __half* gradInput, __half* gradBias, int c, int batchSize);
extern "C" void ClipGrads(__half* grad, int size);
extern "C" void Scale(__half* data, int size, float scale);
extern "C" void LeakyRelu(__half* data, int size, float negativeSlope);
extern "C" void LeakyReluBackward(__half* gradient, const __half* data, int size, float negativeSlope);
extern "C" void SigmoidForward(__half* data, int numSigmoidOutputs, int batchSize, int outputSize);
extern "C" void SigmoidBackward(__half* grad, const __half* data, int numSigmoidOutputs, int batchSize, int outputSize);
extern "C" void LayerNormForward(__half* output, const __half* data, const float* gamma, const float* beta, float* mean, float* variance, int N, int C, int HW, float epsilon);
extern "C" void LayerNormBackward(__half* gradIn, const __half* gradOut, const __half* data, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW, float epsilon);
extern "C" bool IsnanHalf(__half* data, int size);
extern "C" void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, int size, float scale);
extern "C" void BCEGAIL(__half* gradient, const __half* predictions, const __half* discOutput, int batchSize, int numCtrls);
extern "C" void GAILGradient(__half* gradients, const __half* predictions, const __half* discOutput, const float* expertActions, int batchSize, int numCtrls, int numButs, float lambda, float entropyCoeff, float scale, float clip);