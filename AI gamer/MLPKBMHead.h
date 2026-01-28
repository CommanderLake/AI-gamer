#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <vector>
class MLPKBMHead final : public Layer{
public:
	MLPKBMHead(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int embedSize, const char* layerName, bool train, float weightDecay, int gradAccumLength);
	~MLPKBMHead() override;
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
	cudnnHandle_t cudnn_;
	cublasHandle_t cublas_;
	int batchSize_, embedSize_;
	float weightDecay_;
	int gradAccumLength_;
	const float one_ = 1.0f;
	std::vector<Layer*> buttonLayers_;
	std::vector<Layer*> axisLayers_;
	__half* predictions_ = nullptr;
};