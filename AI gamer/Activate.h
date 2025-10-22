#pragma once
#include "Layer.h"
#include <cudnn.h>
class Activate final : public Layer{
public:
	Activate(cudnnHandle_t cudnnHandle, cudnnActivationMode_t mode, double coef, int batchSize, int channels, int height, int width, const char* layerName);
	~Activate() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	cudnnHandle_t cudnnHandle_;
	cudnnActivationDescriptor_t activDesc_;
	int batchSize_, outC_, outHeight_, outWidth_;
	__half *dataIn_ = nullptr, *dataOut_ = nullptr;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
};