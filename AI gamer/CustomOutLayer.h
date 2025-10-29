#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <vector>
class CustomOutLayer : public Layer{
public:
	CustomOutLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int seqLength, int tokens, int embedDim, const char* layerName, bool train, float weightDecay, int gradAccumLength);
	~CustomOutLayer() override;
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
	cudnnTensorDescriptor_t inDesc_;
	int batchSize_, seqLength_, inC_;
	int tokens_;
	int embedDim_;
	int fullFeatureSize_;
	const float alpha_ = 1.0f;
	int gradAccumLength_;
	std::vector<Layer*> buttonLayers_;
	std::vector<Layer*> axisLayers_;
	__half* predictions_ = nullptr;
	__half* upstreamGrad_ = nullptr;
};