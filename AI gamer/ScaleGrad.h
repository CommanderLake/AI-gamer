#pragma once
#include "Layer.h"
#include <cuda_fp16.hpp>
#include <cudnn.h>
class ScaleGrad final : public Layer{
public:
	ScaleGrad(cudnnTensorDescriptor_t outDesc, float scale);
	~ScaleGrad() override;
	__half* forward(__half* data) override;
	__half* backward(__half* grad) override;
	float scale_;
};