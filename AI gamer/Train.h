#pragma once
#include "Viewer.h"
class NN;
class Train{
public:
	Train();
	~Train();
	void Allocate(int batchSize, int stateSize);
	void Free();
	int Train::TrainBatch(NN* nn, const StateBatch* sb, bool smoothLoss, float lr, int batchIndex, int epochBatchCount);
	void TrainModel(int width, int height);
	float lossButs_ = 0.0f;
	float lossAxes_ = 0.0f;
	float emaLossButs_ = 0.0f;
	float emaLossAxes_ = 0.0f;
	unsigned char* dStateBatchBytes = nullptr;
	__half* dStateBatchHalf = nullptr;
	float* hTargetBatchFloat = nullptr;
	float* dTargetBatchFloat = nullptr;
	size_t stateHalfStride_ = 0;
	size_t stateHalfCount_ = 0;
	__half* dGradient_ = nullptr;
};