#pragma once
#include "Layer.h"
#include <cuda_fp16.hpp>
class Swish final : public Layer{
public:
	explicit Swish(int size, const char* layerName);
	~Swish() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	__half* data_ = nullptr;
};