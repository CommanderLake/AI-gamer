#pragma once
#include "Layer.h"
class TransposeLayer final : public Layer{
public:
	TransposeLayer(cudnnHandle_t cudnnHandle, int batchSize, int channels, int height, int width, cudnnTensorFormat_t inFormat, cudnnTensorFormat_t outFormat, const char* layerName);
	~TransposeLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t inDesc_, outDesc_;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	const float alpha_ = 1.0f;
	const float beta_ = 0.0f;
};