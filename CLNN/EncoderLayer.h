#pragma once
#include "Layer.h"
#include <vector>
class __declspec(dllexport) EncoderLayer final : public Layer{
public:
	EncoderLayer(int batchSize, int tokens, int embedDim, int ffDim, int numHeads, std::string layerName, bool train, float weightDecay, int gradAccumLength);
	~EncoderLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float lr) override;
	void SaveParameters(std::ofstream& file, unsigned char* buffer) override;
	void LoadParameters(std::ifstream& file, unsigned char* buffer) override;
	void SaveOptimizerState(std::ofstream& file, unsigned char* buffer) override;
	void LoadOptimizerState(std::ifstream& file, unsigned char* buffer) override;
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
	void SetTrain(bool enable) override;
	void CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks) override;
private:
	int batchSize_, tokens_, embedDim_, ffDim_;
	int gradAccumLength_;
	std::vector<Layer*> layers_;
	__half* residualGrad_ = nullptr;
	const float mixFwd_ = 1.0f;
	const float mixBwd_ = 1.0f;
};
