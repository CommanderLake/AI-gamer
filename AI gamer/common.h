#pragma once
#include "ThreadPool.h"
#include <cudnn.h>
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
		keyStates = static_cast<unsigned short*>(malloc(batchSize*sizeof(unsigned short)));
		mouseDeltaX = static_cast<int*>(malloc(batchSize*sizeof(int)));
		mouseDeltaY = static_cast<int*>(malloc(batchSize*sizeof(int)));
		if(!keyStates || !mouseDeltaX || !mouseDeltaY){ throw std::bad_alloc(); }
		checkCUDA(cudaMallocHost(reinterpret_cast<void**>(&stateData), stateSize*batchSize));
	}
	~StateBatch(){
		free(keyStates);
		free(mouseDeltaX);
		free(mouseDeltaY);
		cudaFreeHost(stateData);
	}
};
struct ConvolutionAlgorithms{
	cudnnConvolutionFwdAlgo_t fwdAlgo;
	cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
	cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
	size_t workspaceSize;
};
extern std::vector<std::string> trainDataInFiles;
extern std::unordered_map<std::string, std::vector<std::streampos>> fileRecordIndex;
extern std::size_t stateSize_;
extern ThreadPool threadPool;
void LoadBatch(StateBatch* batch, int batchSize);
void LoadBatch3D(StateBatch* batch, int seqLength, int batchSize);
void LoadBatchLSTM(StateBatch* batch, int seqLength, int batchSize);
ConvolutionAlgorithms GetConvolutionAlgorithms(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, bool isTraining);
template <typename T>
void CUDAMallocZero(T** ptr, size_t size){
	checkCUDA(cudaMalloc(reinterpret_cast<void**>(ptr), size));
	checkCUDA(cudaMemset(*ptr, 0, size));
}
const std::string trainDataOutFileName("E:\\training_data.bin");
const std::string ckptFileName("E:\\AIGamer.ckpt");
const std::string optFileName("E:\\AIGamer.opt");
const std::string ckptFileNameDisc("E:\\AIGamerDisc.ckpt");
const std::string optFileNameDisc("E:\\AIGamerDisc.opt");
constexpr int numButs_ = 14;
constexpr int numAxes_ = 2;
constexpr int numCtrls_ = numButs_ + numAxes_;
extern unsigned char keyMap[14];
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
extern "C" void MseLoss2(const __half* dPredictions, const float* dTargets, int numButs, int numCtrls, int batchSize, float* butLoss, float* axesLoss);
extern "C" void ConvertAndNormalize(__half* output, const unsigned char* input, size_t size);
extern "C" void UnConvertAndUnNormalize(unsigned char* output, const __half* input, size_t size);
extern "C" void ConvertFloatToHalf(float* src, __half* dst, size_t n);
extern "C" void ConvertHalfToFloat(__half* src, float* dst, size_t n);
extern "C" void HeInit(__half* weightHalf, int numWeights, float fanIn);
extern "C" void SGDHalf(__half* param, const __half* grads, int size, float learningRate, float weightDecay);
extern "C" void SGDFloat(float* param, const float* grads, int size, float learningRate, float weightDecay);
extern "C" void AdamWHalf(__half* params, const __half* grads, __half* m, __half* v, float lr, int t, float weightDecay, int size);
extern "C" void AdamWFloat(float* params, const float* grads, float* m, float* v, float learningRate, int t, float weightDecay, int size);
extern "C" void AdanHalf(__half* params, const __half* grads, __half* m, __half* v, __half* n, __half* velocity, float learningRate, int t, float weightDecay, int size);
extern "C" void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, int size);
extern "C" void BiasGradient(const __half* gradInput, __half* gradBias, int c, int batchSize);
extern "C" void LeakyReluForward(__half* data, int size, float negativeSlope);
extern "C" void LeakyReluBackward(__half* grad, const __half* data, int size, float negativeSlope);
extern "C" void SwishForward(__half* data, int size);
extern "C" void SwishBackward(__half* grad, const __half* data, int size);
extern "C" void SigmoidForward(__half* data, int numCtrls, int numButs, int size);
extern "C" void SigmoidBackward(__half* grad, const __half* data, int numCtrls, int numButs, int size);
extern "C" void LayerNormForward(__half* output, const __half* data, const float* gamma, const float* beta, float* mean, float* variance, int N, int C, int HW);
extern "C" void LayerNormBackward(__half* grad, const __half* data, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW);
extern "C" bool IsnanHalf(const __half* data, int size);
extern "C" void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, int size, float scale);
extern "C" void DiscriminatorGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, int size, int numCtrls, int numButs, float binaryScale, float continuousScale, float clip);
extern "C" void GAILGradient(__half* gradients, const __half* predictions, const __half* discOutput, const float* expertActions, int batchSize, int numCtrls, int numButs, float lambda, float entropyCoeff, float butScale, float axiScale, float clip);
extern "C" void ComputeAttention(const __half* inData, const __half* attentionMap, __half* outData, int inC, int attC, int inH, int inW, int size);
extern "C" void ComputeQueryKeyGrad(const __half* gradAttention, const __half* queryMap, const __half* keyMap, __half* gradQuery, __half* gradKey, int inC, int attC, int inH, int inW, int size);
extern "C" void ApplyAttention(const __half* valueMap, const __half* attentionScores, __half* output, int inC, int attC, int inH, int inW, int size);
extern "C" void ApplyAttentionBackward(const __half* grad, const __half* queryMap, const __half* attentionMap, __half* gradQueryMap, __half* gradAttentionMap, int inC, int attC, int inH, int inW, int size);
extern "C" void SpatialAttentionForward(const __half* __restrict__ inData, const __half* __restrict__ keyWeights, const __half* __restrict__ queryWeights, const __half* __restrict__ valueWeights, __half* __restrict__ keyMap,
	__half* __restrict__ queryMap, __half* __restrict__ valueMap, __half* __restrict__ attentionScores, __half* __restrict__ outData, int batchSize, int inC, int attC, int inH, int inW);
extern "C" void SpatialAttentionBackward(const __half* __restrict__ inData, const __half* __restrict__ keyWeights, const __half* __restrict__ queryWeights, const __half* __restrict__ valueWeights, const __half* __restrict__ keyMap,
	const __half* __restrict__ queryMap, const __half* __restrict__ valueMap, const __half* __restrict__ attentionScores, const __half* __restrict__ inGrad, __half* __restrict__ outGrad,
	__half* __restrict__ keyWeightsGrad, __half* __restrict__ queryWeightsGrad, __half* __restrict__ valueWeightsGrad, int batchSize, int inC, int attC, int inH, int inW);