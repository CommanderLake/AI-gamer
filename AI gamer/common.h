#pragma once
#include "ThreadPool.h"
#include <string>
#include <iostream>
#include <cuda_runtime_api.h>
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
extern std::vector<std::string> trainDataFiles;
extern std::string valDataFile;
extern std::string trainDataOutFileName;
extern std::string ckptFileName;
extern std::string optFileName;
extern std::vector<RecordIndex> trainRecordIndices;
extern std::vector<RecordIndex> valRecordIndices;
extern ThreadPool threadPool;
constexpr int TGT_STATE_WIDTH_ = 320;
constexpr int NUM_BUTS_ = 14;
constexpr int NUM_AXES_ = 2;
constexpr int NUM_CTRLS_ = NUM_BUTS_ + NUM_AXES_;
constexpr float AXIS_SCALE_ = 1024.0f;
constexpr float COMP_SCALE_ = 1024.0f;
inline float CompressAxisDelta(const float delta){
	return std::copysign(std::log1pf(std::fabs(delta)/COMP_SCALE_), delta)/AXIS_SCALE_;
}
inline float DecompressAxisDelta(const float encoded){
	return std::copysign(COMP_SCALE_*std::expm1f(std::fabs(encoded)*AXIS_SCALE_), encoded);
}
extern unsigned char keyMap[14];
void LoadBatch(StateBatch* batch, int batchSize, int stateSize, bool validation);
void LoadBatchFromVector(const std::vector<StateSingle*>& states, StateBatch* batch, int batchSize, int stateSize);
void ResetLoadBatchFailureCount();
int GetLoadBatchFailureCount();
void ShuffleBatchOrder(bool validation);
void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, int numCtrls, int numButs, int size);
void GetPrediction(const __half* predBatch, float* prediction, int numCtrls, int batchSize);