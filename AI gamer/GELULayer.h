#pragma once
#include "Layer.h"
class GELU final : public Layer{
public:
	GELU(int batchSize, int channels, int height, int width, const char* layerName);
	~GELU() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	int batchSize_, outC_, outHeight_, outWidth_;
	__half* dataOut_ = nullptr;
	__half* data_ = nullptr;
};