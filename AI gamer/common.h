#pragma once
#include "ThreadPool.h"
#include "WeightInitMethod.h"
#include <cudnn.h>
#include <string>
#include <iostream>
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
extern std::vector<std::string> trainDataFiles;
extern std::string valDataFile;
extern std::string trainDataOutFileName;
extern std::string ckptFileName;
extern std::string optFileName;
extern std::vector<RecordIndex> trainRecordIndices;
extern std::vector<RecordIndex> valRecordIndices;
extern ThreadPool threadPool;
void LoadBatch(StateBatch* batch, int batchSize, int stateSize, bool validation);
void LoadBatchLSTM(StateBatch* batch, int batchSize, int seqLength, int stateSize, bool validation);
void LoadBatchFromVector(const std::vector<StateSingle*>& states, StateBatch* batch, int batchSize, int stateSize);
ConvolutionAlgorithms GetConvolutionAlgorithms(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, bool isTraining);
void OrthogonalInit(__half* output, int rows, int cols, WeightInitMethod method);
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
void SummarizeHalfDevice(const __half* data, size_t size, const char* label);