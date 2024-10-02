#pragma once
#include "Layer.h"
#include <cuda_fp16.hpp>
#include <cudnn.h>
class LeakyReLU final : public Layer{
public:
	explicit LeakyReLU(int size, const char* layerName);
	~LeakyReLU() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	__half* data_ = nullptr;
	float slope_;
};