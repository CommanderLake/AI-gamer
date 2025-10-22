#pragma once
#include "Layer.h"
#include <vector>
class EncoderLayer final : public Layer{
public:
	EncoderLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int ffDim, int numHeads, const char* layerName, bool train, float weightDecay, int gradAccumLength);
	~EncoderLayer() override;
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
	void SetDropout(bool enable) override;
private:
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	int batchSize_, tokens_, embedDim_, ffDim_;
	int gradAccumLength_;
	std::vector<Layer*> layers_;
	const float mixFwd_ = 1.0f;
	const float mixBwd_ = 1.0f;
};