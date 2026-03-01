#pragma once
#include "common.h"
#include "Layer.h"
#include <vector>
class NN{
public:
	NN(cudnnHandle_t cudnnHandle, int w, int h, bool train);
	~NN();
	__half* Forward(__half* data);
	__half* Backward(__half* grad);
	void UpdateParams(float lr);
	void SaveModel(const std::string& filename);
	void SaveOptimizerState(const std::string& filename);
	void SetTrain(bool enable);
	cudnnHandle_t cudnn_;
	std::vector<Layer*> layers_;
	int batchSize_;
	int stateSize_;
	int inWidth_ = 0, inHeight_ = 0;
	size_t maxBufferSize_ = 0;
	int gradAccumLength_;
};