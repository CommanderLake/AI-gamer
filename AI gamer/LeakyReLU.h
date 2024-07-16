#pragma once
#include "Layer.h"
#include <cuda_fp16.hpp>
#include <cudnn.h>
class LeakyReLU final : public Layer{
public:
	LeakyReLU(cudnnTensorDescriptor_t outDesc);
	~LeakyReLU() override;
	__half* forward(__half* data) override;
	__half* backward(__half* grad) override;
	__half* data_ = nullptr;
};