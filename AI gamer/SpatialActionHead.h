#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <vector>
class SpatialActionHead : public Layer{
public:
	SpatialActionHead(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int patchRows, int patchCols, int embedSize, const char* layerName, bool train, float weightDecay, int gradAccumLength);
	~SpatialActionHead() override;
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
	cudnnTensorDescriptor_t neckDesc_;
	int batchSize_, nTokens_, tokensWithCls_, embedSize_;
	int patchRows_, patchCols_;
	int sharedHeight_, sharedWidth_;
	float weightDecay_;
	int gradAccumLength_;
	const float one_ = 1.0f;
	int trunkC1_, trunkC2_, sharedOutC_;
	std::vector<Layer*> sharedLayers_;
	std::vector<Layer*> buttonLayers_;
	std::vector<Layer*> axisLayers_;
	std::vector<Layer*> classLayers_;
	__half* classTokens_ = nullptr;
	__half* patchTokens_ = nullptr;
	__half* classNeckGrad_ = nullptr;
	__half* upstreamGrad_ = nullptr;
	__half* predictions_ = nullptr;
};