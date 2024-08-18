#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <vector>
class CustomOutLayer : public Layer{
public:
	CustomOutLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, cudnnTensorDescriptor_t outDesc, int batchSize, int inputSize, int hiddenSize, int buttons, int axes, const char* layerName, bool train);
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float learningRate) override;
	void SaveParameters(std::ofstream& file, float* buffer) override;
	void LoadParameters(std::ifstream& file, float* buffer) override;
	void SaveOptimizerState(std::ofstream& file, float* buffer) override;
	void LoadOptimizerState(std::ifstream& file, float* buffer) override;
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	int batchSize_;
	std::vector<Layer*> buttonLayers_;
	std::vector<Layer*> axisLayers_;
	__half* outData_;
	size_t buttons_;
	size_t axes_;
	const float alpha = 1.0f;
};