#pragma once
#include "Layer.h"
class SkipLayer final : public Layer{
	SkipLayer(cudnnHandle_t cudnnHandle, __half* inB, int batchSize, int channels, int height, int width, const char* layerName);
	~SkipLayer() override;
public:
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	cudnnHandle_t cudnnHandle_;
	cudnnOpTensorDescriptor_t opDesc_;
	cudnnTensorDescriptor_t inDesc_;
	__half* inB_ = nullptr;
	__half* outData_ = nullptr;
	const float alpha = 1.0f;
	const float beta = 0.0f;
};