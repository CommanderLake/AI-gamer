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
	ResFCLayer(cudaStream_t cudaStream, cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inC, int outC, const char* layerName, bool train, float weightDecay);
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
	cudaStream_t cudaStream_;
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t inDesc_;
	int batchSize_;
	std::vector<Layer*> layers_;
	FCLayer* residue_;
	Activate* resAct_;
	const float blendFwd = 0.5f;
	const float blendBwd = 0.1f;
};