#pragma once
#include "Layer.h"
#include <cuda_fp16.hpp>
#include <cudnn.h>
class Activate final : public Layer{
public:
	Activate(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, cudnnActivationMode_t mode, double coef, const char* layerName);
	~Activate() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	cudnnHandle_t cudnnHandle_;
	cudnnActivationDescriptor_t activDesc_;
	cudnnTensorDescriptor_t gradDesc_;
	__half* data_ = nullptr;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
};