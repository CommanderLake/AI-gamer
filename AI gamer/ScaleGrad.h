#pragma once
#include "Layer.h"
#include <cuda_fp16.hpp>
#include <cudnn.h>
class ScaleGrad final : public Layer{
public:
	ScaleGrad(cudnnTensorDescriptor_t outDesc, float scale);
	~ScaleGrad() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	float scale_;
};