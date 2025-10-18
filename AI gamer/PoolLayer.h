#pragma once
#include "Layer.h"
class PoolLayer final : public Layer{
public:
	PoolLayer(cudnnHandle_t cudnnHandle, cudnnPoolingMode_t mode, int batchSize, int channels, int* height, int* width, int poolH, int poolW, int strideH, int strideW, const char* layerName, bool train);
	~PoolLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t inDesc_;
	cudnnPoolingDescriptor_t poolDesc_;
	__half* inData_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	const float alpha_ = 1.0f;
	const float beta0_ = 0.0f;
	int inHeight_, inWidth_;
	int batchSize_, outC_, outHeight_, outWidth_;
	int inNCHW_;
};