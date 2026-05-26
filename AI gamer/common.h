#pragma once
#include "ThreadPool.h"
#include <string>
#include <iostream>
#include <vector>
#include <cmath>
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
struct StateBatchSequence{
	int batchSize;
	int sequenceLength;
	int stateSize;
	InputState* sequenceInputStates;
	InputState* targetInputStates;
	unsigned char* sequenceStateData = nullptr;
	explicit StateBatchSequence(const int batchSize, const int sequenceLength, const int stateSize) : batchSize(batchSize), sequenceLength(sequenceLength), stateSize(stateSize){
		if(cudaMallocHost(reinterpret_cast<void**>(&sequenceStateData), static_cast<size_t>(stateSize)*batchSize*sequenceLength)!=cudaSuccess){
			throw std::runtime_error("Failed to allocate pinned memory with cudaMallocHost for sequence state data");
		}
		sequenceInputStates = new InputState[batchSize*sequenceLength];
		targetInputStates = new InputState[batchSize];
	}
	~StateBatchSequence(){
		delete[] sequenceInputStates;
		delete[] targetInputStates;
		if(sequenceStateData){ cudaFreeHost(sequenceStateData); }
	}
};
struct RecordIndex{
	const std::string* fileName;
	std::streampos position;
};
struct SequenceRecordIndex{
	const std::string* fileName;
	std::streampos startPosition;
};
struct SequenceSamplingConfig{
	int length = 1;
	int stride = 1;
	int targetOffset = 0;
};
extern std::vector<std::string> trainDataFiles;
extern std::string valDataFile;
extern std::string trainDataOutFileName;
extern std::string ckptFileName;
extern std::string optFileName;
extern std::vector<RecordIndex> trainRecordIndices;
extern std::vector<RecordIndex> valRecordIndices;
extern std::vector<SequenceRecordIndex> trainSequenceRecordIndices;
extern std::vector<SequenceRecordIndex> valSequenceRecordIndices;
extern SequenceSamplingConfig sequenceSamplingConfig;
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
void LoadBatchSequence(StateBatchSequence* batch, int batchSize, int stateSize, bool validation);
void ConfigureSequenceSampling(int length, int stride, int targetOffset);
void RebuildSequenceIndices();
void ResetLoadBatchFailureCount();
int GetLoadBatchFailureCount();
void ShuffleBatchOrder(bool validation);
void SelectLastTemporalFrame(const __half* input, __half* output, int batchSize, int temporalLength, int featureSize);
void ExpandTemporalOutputs(const __half* input, __half* output, int batchSize, int temporalLength, int featureSize);
void ReduceTemporalGradients(const __half* inputGrad, __half* reducedGrad, int batchSize, int temporalLength, int featureSize);
void ScatterLastTemporalFrameGrad(const __half* input, __half* output, int batchSize, int temporalLength, int featureSize);
void MergeOutputs(__half* predOut, const __half* buttonData, const __half* axisData, int numCtrls, int numButs, int size);
void GetPrediction(const __half* predBatch, float* prediction, int numCtrls, int batchSize);
