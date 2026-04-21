#pragma once
#include "Layer.h"
class __declspec(dllexport) GlobalPoolLayer final : public Layer{
public:
	GlobalPoolLayer(int batchSize, int nTokens, int embedSize, int numQueries, std::string layerName, bool train);
	~GlobalPoolLayer() override;
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
	int batchSize_, nTokens_, embedSize_, numQueries_;
	float invSqrtDim_;
	float weightDecay_ = 0.0f;
	int t_ = 1;
	__half* inData_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	__half* gradQuery_ = nullptr;
	__half* mQuery_ = nullptr;
	__half* vQuery_ = nullptr;
	float* attnWeights_ = nullptr;
	float* scratchBuffer_ = nullptr;
	float* batchSums_ = nullptr;
};