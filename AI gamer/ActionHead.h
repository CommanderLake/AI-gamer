#pragma once
#include "Layer.h"
#include <vector>
#include <cuda_fp16.h>
enum class TemporalOutputPolicy{
	LastFrameBroadcast
};
class ActionHead final : public Layer{
public:
	ActionHead(int batchSize, int patchRows, int patchCols, int embedSize, std::string layerName, bool train, float weightDecay, int gradAccumLength, int temporalLength = 1, TemporalOutputPolicy outputPolicy = TemporalOutputPolicy::LastFrameBroadcast);
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
	int batchSize_, nTokens_, embedSize_;
	int temporalLength_ = 1;
	int baseBatchSize_ = 0;
	int patchRows_, patchCols_;
	int inC_;
	int outFeatureSize_ = 0;
	TemporalOutputPolicy outputPolicy_;
	float weightDecay_;
	int gradAccumLength_;
	const float one_ = 1.0f;
	__half* temporalInput_ = nullptr;
	__half* temporalOut_ = nullptr;
	__half* temporalGradReduced_ = nullptr;
	__half* temporalGradOut_ = nullptr;
	std::vector<Layer*> layers_;
};
