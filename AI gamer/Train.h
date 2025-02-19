#pragma once
#include "Viewer.h"
class NN;
class Train{
public:
	Train();
	~Train();
	void Allocate(int batchSize, int seqLength, int stateSize);
	void Free();
	int TrainBatch(NN* nn, const StateBatch* sb, bool smoothLoss, float lr);
	void TrainModel(int width, int height);
	void TuneModel(NN* nn, const std::vector<StateSingle*>& states, int epochs, float lr);
	float lossButs_ = 0.0f;
	float lossAxes_ = 0.0f;
	float emaLossButs_ = 0.0f;
	float emaLossAxes_ = 0.0f;
	unsigned char* dStateBatchBytes = nullptr;
	__half* dStateBatchHalf = nullptr;
	float* hTargetBatchFloat = nullptr;
	float* dTargetBatchFloat = nullptr;
	__half* dy_ = nullptr;
};