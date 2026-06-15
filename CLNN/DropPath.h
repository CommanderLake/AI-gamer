#pragma once
#include "Layer.h"
class __declspec(dllexport) DropPath final : public Layer{
public:
	DropPath(float dropRate, int batchSize, int elementsPerBatch, std::string layerName, bool train);
	~DropPath() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	__half* ReplayForward(__half* data);
	void SetTrain(bool enable) override;
	float dropRate_;
	float keepProb_;
	int batchSize_;
	int elementsPerBatch_;
	float* mask_ = nullptr;
};
