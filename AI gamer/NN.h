#pragma once
#include "common.h"
#include "Layer.h"
#include <vector>
class NN{
public:
	NN(int w, int h, bool train);
	~NN();
	__half* Forward(__half* data);
	__half* Backward(__half* grad);
	void UpdateParams(float lr);
	void SaveModel(const std::string& filename);
	void SaveOptimizerState(const std::string& filename);
	void SetTrain(bool enable);
	std::vector<Layer*> layers_;
	int batchSize_;
	int stateSize_;
	int inWidth_ = 0, inHeight_ = 0;
	size_t maxBufferSize_ = 0;
	int gradAccumLength_;
	std::vector<AdamWHalfTask> adamWHalfTasks_;
	std::vector<AdamWFloatTask> adamWFloatTasks_;
	AdamWHalfTask* dAdamWHalfTasks_ = nullptr;
	AdamWFloatTask* dAdamWFloatTasks_ = nullptr;
	int totalAdamWHalfSize_ = 0;
	int totalAdamWFloatSize_ = 0;
	int adamWStep_ = 1;
	int accumStep_ = 0;
	void CollectAdamWTasks();
};