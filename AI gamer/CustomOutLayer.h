#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <vector>
class CustomOutLayer : public Layer{
public:
	CustomOutLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, cudnnTensorDescriptor_t outDesc, int batchSize, int inputSize, int hiddenSize, const char* layerName, bool train);
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float learningRate) override;
	void SaveParameters(std::ofstream& file, unsigned char* buffer) override;
	void LoadParameters(std::ifstream& file, unsigned char* buffer) override;
	void SaveOptimizerState(std::ofstream& file, unsigned char* buffer) override;
	void LoadOptimizerState(std::ifstream& file, unsigned char* buffer) override;
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	int batchSize_;
	std::vector<Layer*> buttonLayers_;
	std::vector<Layer*> axisLayers_;
	__half* outData_;
	const float alpha = 1.0f;
};