#pragma once
#include "Layer.h"
#include <cuda_fp16.hpp>
class Swish final : public Layer{
public:
	explicit Swish(int batchSize, int channels, int height, int width, const char* layerName);
	~Swish() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	int batchSize_, outC_, outHeight_, outWidth_;
	__half* data_ = nullptr;
};