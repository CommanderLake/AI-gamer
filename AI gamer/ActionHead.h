#pragma once
#include "Layer.h"
#include <vector>
#include <cuda_fp16.h>
class ActionHead final : public Layer{
public:
	ActionHead(cudnnHandle_t cudnnHandle, int batchSize, int patchRows, int patchCols, int embedSize, std::string layerName, bool train, float weightDecay, int gradAccumLength);
	~ActionHead() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float learningRate) override;
	void SaveParameters(std::ofstream& file, unsigned char* buffer) override;
	void LoadParameters(std::ifstream& file, unsigned char* buffer) override;
	void SaveOptimizerState(std::ofstream& file, unsigned char* buffer) override;
	void LoadOptimizerState(std::ifstream& file, unsigned char* buffer) override;
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
	void SetTrain(bool enable) override;
	void CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks) override;
	cudnnHandle_t cudnn_;
	int batchSize_, nTokens_, embedSize_;
	int patchRows_, patchCols_;
	int inC_;
	float weightDecay_;
	int gradAccumLength_;
	const float one_ = 1.0f;
	std::vector<Layer*> layers_;
};