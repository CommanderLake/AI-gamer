#pragma once
#include "ThreadPool.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <string>
#include <iostream>
const char* cublasGetErrorString(cublasStatus_t status);
#define checkCUBLAS(status) { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "\ncuBLAS error: " << cublasGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuBLAS error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cublasGetErrorString(status)); \
    } \
}

#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "\ncuDNN error: " << cudnnGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("cuDNN error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudnnGetErrorString(status)); \
    } \
}

#define checkCUDA(status) { \
    if (status != cudaSuccess) { \
        std::cerr << "\nCUDA error: " << cudaGetErrorString(status) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        throw std::runtime_error("CUDA error at " + std::string(__FILE__) + ":" + std::to_string(__LINE__) + " - " + cudaGetErrorString(status)); \
    } \
}
enum class InferMode{
	Off,
	On,
	Correct,
	Tune
};
struct __half;
struct InputState{
	unsigned int keyStates;
	int deltaX;
	int deltaY;
};
struct StateSingle{
	InputState inputState;
	unsigned char* stateData = nullptr;
	explicit StateSingle(const InputState& inputState, const unsigned char* data, const size_t stateSize, bool fromGPU): inputState(inputState){
		stateData = static_cast<unsigned char*>(_mm_malloc(stateSize, 32));
		if(!stateData){ throw std::bad_alloc(); }
		if(fromGPU) cudaMemcpy(stateData, data, stateSize, cudaMemcpyDeviceToHost);
		else memcpy(stateData, data, stateSize);
	}
	~StateSingle(){ if(stateData){ _mm_free(stateData); } }
};
struct StateBatch{
	int batchSize;
	int stateSize;
	InputState* inputStates;
	unsigned char* stateData = nullptr;
	explicit StateBatch(const int batchSize, const int stateSize) : batchSize(batchSize), stateSize(stateSize){
		if(cudaMallocHost(reinterpret_cast<void**>(&stateData), stateSize*batchSize)!=cudaSuccess){
			throw std::runtime_error("Failed to allocate pinned memory with cudaMallocHost");
		}
		inputStates = new InputState[batchSize];
	}
	~StateBatch(){
		delete[] inputStates;
		if(stateData){ cudaFreeHost(stateData); }
	}
};
struct RecordIndex{
	const std::string* fileName;
	std::streampos position;
};
struct ConvolutionAlgorithms{
	cudnnConvolutionFwdAlgo_t fwdAlgo;
	cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
	cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
	size_t workspaceSize;
};
enum WeightInitMethod{
	He, Xavier, Orthogonal
};
extern std::vector<std::string> trainDataFiles;
extern std::string valDataFile;
extern std::vector<RecordIndex> trainRecordIndices;
extern std::vector<RecordIndex> valRecordIndices;
extern ThreadPool threadPool;
void LoadBatch(StateBatch* batch, int batchSize, int stateSize, bool validation);
void LoadBatchLSTM(StateBatch* batch, int batchSize, int seqLength, int stateSize, bool validation);
void LoadBatchFromVector(const std::vector<StateSingle*>& states, StateBatch* batch, int batchSize, int stateSize);
ConvolutionAlgorithms GetConvolutionAlgorithms(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, bool isTraining);
void OrthogonalInit(__half* output, int rows, int cols);
template <typename T>
void CUDAMallocZero(T** ptr, size_t size){
	checkCUDA(cudaMalloc(reinterpret_cast<void**>(ptr), size));
	checkCUDA(cudaMemset(*ptr, 0, size));
}
const std::string trainDataOutFileName("I:\\TrainingData.bin");
const std::string ckptFileName("I:\\AIGamer.ckpt");
const std::string optFileName("I:\\AIGamer.opt");
constexpr int NUM_BUTS_ = 14;
constexpr int NUM_AXES_ = 2;
constexpr int NUM_CTRLS_ = NUM_BUTS_ + NUM_AXES_;
constexpr int TGT_STATE_WIDTH_ = 320;
extern unsigned char keyMap[14];
void ClearScreen(char fill = ' ');
int ConvertSmVer2Cores(int major, int minor);
void HalfToFloatAsm(float* dst, __half* src, int count);
void FloatToHalfAsm(float* src, __half* dst, int count);
void PrintDataHalfDevice(const __half* data, size_t size, const char* label);
void PrintDataFloatDevice(const float* data, size_t size, const char* label);
void PrintDataFloatHost(const float* data, size_t size, const char* label);
void PrintDataCharHost(const unsigned char* data, size_t size, const char* label);
extern "C" void InitCUDA();
extern "C" float MseLoss(const __half* dPredictions, const float* dTargets, int size);
extern "C" void MseLoss2(const __half* dPredictions, const float* dTargets, int numButs, int numCtrls, int batchSize, float* butLoss, float* axesLoss);
extern "C" void BlockShiftHalf(__half* hPtr, int shiftBy, int blocksToShift);
extern "C" void ConvertByteToHalf(const unsigned char* input, __half* output, size_t size, bool normalize);
extern "C" void ConvertHalfToByte(const __half* input, unsigned char* output, size_t size, bool normalize);
extern "C" void ConvertFloatToHalf(const float* input, __half* output, size_t size);
extern "C" void ConvertHalfToFloat(const __half* input, float* output, size_t size);
extern "C" void ConvertFloatToHalfScale(__half* halfWeights, const float* weights, size_t size, float scale);
extern "C" void WeightInit(__half* weightHalf, int numWeights, int fanIn, int fanOut, WeightInitMethod method);
extern "C" void SGDHalf(__half* params, const __half* grads, int size, float learningRate, float weightDecay);
extern "C" void SGDFloat(float* params, const float* grads, int size, float learningRate, float weightDecay);
extern "C" void AdamWHalf(__half* params, const __half* grads, __half* m, __half* v, float lr, int t, float weightDecay, int size);
extern "C" void AdamWFloat(float* params, const float* grads, float* m, float* v, float learningRate, int t, float weightDecay, int size);
extern "C" void Gradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, float clip, int size);
extern "C" void SplitGradient(__half* dGradient, const __half* dPredictions, const float* dTargets, float clip, int size, int numCtrls, int numButs, int batchSize);
extern "C" void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, int numCtrls, int numButs, int size);
extern "C" void GetPrediction(const __half* predBatch, float* prediction, int numCtrls, int batchSize);
extern "C" void LeakyReluForward(const __half* dataIn, __half* dataOut, int size, float negativeSlope, cudaStream_t stream = nullptr);
extern "C" void LeakyReluBackward(__half* grad, const __half* dataIn, int size, float negativeSlope, cudaStream_t stream = nullptr);
extern "C" void SwishForward(const __half* dataIn, __half* outData, int size, cudaStream_t stream = nullptr);
extern "C" void SwishBackward(__half* grad, const __half* dataIn, int size, cudaStream_t stream = nullptr);
extern "C" void SigmoidForward(const __half* dataIn, __half* dataOut, int numCtrls, int numButs, int size, cudaStream_t cudaStream = nullptr);
extern "C" void SigmoidBackward(__half* grad, const __half* dataIn, int numCtrls, int numButs, int size, cudaStream_t cudaStream = nullptr);
extern "C" void GELUForward(const __half* dataIn, __half* dataOut, int size, cudaStream_t stream = nullptr);
extern "C" void GELUBackward(__half* grad, const __half* dataIn, int size, cudaStream_t stream = nullptr);
extern "C" void LayerNormForward(__half* dataOut, const __half* dataIn, const float* gamma, const float* beta, float* mean, float* variance, int N, int C, int HW);
extern "C" void LayerNormBackward(__half* grad, const __half* dataIn, const float* gamma, float* gradGamma, float* gradBeta, const float* mean, const float* variance, int N, int C, int HW);
extern "C" bool IsnanHalf(const __half* data, int size);
extern "C" void BCEGradient(__half* dGradient, const __half* dPredictions, const __half* dTargets, int size, float scale);
extern "C" void ComputeAttention(const __half* queryMap, const __half* keyMap, __half* attentionScores, int inC, int attC, int inH, int inW);
extern "C" void ApplyAttention(const __half* valueMap, const __half* attentionScores, __half* output, int inC, int attC, int inH, int inW);
extern "C" void ApplyAttentionBackward(const __half* gradIn, const __half* valueMap, const __half* attentionScores, __half* gradValue, __half* gradAttention, int inC, int attC, int inH, int inW);
extern "C" void ComputeQueryKeyGrad(const __half* gradAttention, const __half* queryMap, const __half* keyMap, __half* gradQuery, __half* gradKey, int N, int inC, int attC, int inH, int inW);
extern "C" void SpatialSoftmaxHalf(const __half* inData, __half* outData, int N, int C, int H, int W);
extern "C" void SpatialSoftmaxBackwardHalf(const __half* outData, const __half* gradIn, __half* gradOut, int N, int C, int H, int W);
extern "C" void FeatureMapMosaic(const __half* dInput, unsigned char* dOutput, int H, int W, int inC, int mosaicW, int tileW, int tileH, int gridW, cudaStream_t stream = nullptr);
extern "C" void WmmaAttention(const __half* Q, const __half* K, const __half* V, __half* Out, int B, int T, int D, int H);