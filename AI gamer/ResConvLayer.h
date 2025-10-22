#pragma once
#include "Layer.h"
#include "ConvLayer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
#include "Activate.h"
#include "LeakyReLULayer.h"
#include "SwishLayer.h"
class ResConvLayer : public Layer{
public:
	ResConvLayer(cudnnHandle_t cudnnHandle, int batchSize, int inC, int outC, int *inHeight, int *inWidth, const char* layerName, bool train, float weightDecay, int gradAccumLength);
	~ResConvLayer() override;
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
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t inDesc_;
	int batchSize_, outC_, outHeight_, outWidth_;
	int inC_, inHeight_, inWidth_;
	std::vector<Layer*> layers_;
	ConvLayer* residue_;
	Activate* resAct_;
	const float blendFwd = 0.5f;
	const float blendBwd = 0.25f;
	int gradAccumLength_;
};