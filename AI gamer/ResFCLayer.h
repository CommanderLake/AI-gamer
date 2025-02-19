#pragma once
#include "Layer.h"
#include "ConvLayer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
#include "Activate.h"
#include "FCLayer.h"
class ResFCLayer : public Layer{
public:
	ResFCLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inC, int hiddenC, int outC, const char* layerName, bool train, float weightDecay, int gradAccumLength);
	~ResFCLayer() override;
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
	cudaStream_t cudaStream_;
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t inDesc_;
	int batchSize_, inC_, hiddenC_, outC_;
	std::vector<Layer*> layers_;
	FCLayer* residue_;
	Activate* resAct_;
	const float fwdAlpha = 0.2f;
	const float fwdBeta = 0.8f;
	const float bwdAlpha = 0.2f;
	const float bwdBeta = 0.8f;
	int gradAccumLength_;
};