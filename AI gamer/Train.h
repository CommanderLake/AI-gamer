#pragma once
#include "Viewer.h"
class NN;
class Train{
public:
	Train();
	~Train();
	void Allocate(int batchSize, int sequenceLength, int stateSize);
	void Free();
	int TrainBatch(NN* generator, const StateBatch* sb, const int stateSize, bool averageLoss);
	void TrainModel(int width, int height);
	void TuneModel(NN* generator, const std::vector<StateSingle*>& states, int epochs);
	float lossButs_ = 0.0f;
	float lossAxes_ = 0.0f;
	float emaLossButs_ = 0.0f;
	float emaLossAxes_ = 0.0f;
	unsigned char* dStateBatchBytes = nullptr;
	__half* dstateBatchHalf = nullptr;
	float* hTargetBatchFloat = nullptr;
	float* dTargetBatchFloat = nullptr;
	__half* dTargetBatchHalf = nullptr;
	__half* dGeneratorGrad = nullptr;
};