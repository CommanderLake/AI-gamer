#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <vector>
class SpatialActionHead : public Layer{
public:
	SpatialActionHead(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int seqLength, int patchRows, int patchCols, int embedDim, const char* layerName, bool train, float weightDecay, int gradAccumLength);
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
	void SetFineTune(bool enable) override;
	void SetDropout(bool enable) override;
private:
	void UpdateSharedDescriptor();
	cudnnHandle_t cudnn_;
	cublasHandle_t cublas_;
	cudnnTensorDescriptor_t sharedDesc_;
	int ogbs_;
	int batchSize_;
	int seqLength_;
	int nTokens_;
	int patchRows_;
	int patchCols_;
	int embedDim_;
	int sharedHeight_;
	int sharedWidth_;
	float weightDecay_;
	int gradAccumLength_;
	const float alpha_ = 1.0f;
	std::vector<Layer*> sharedLayers_;
	std::vector<Layer*> buttonLayers_;
	std::vector<Layer*> axisLayers_;
	__half* spatialInput_ = nullptr;
	__half* sharedOutput_ = nullptr;
	__half* tokenGrad_ = nullptr;
	__half* predictions_ = nullptr;
};