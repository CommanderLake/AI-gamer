#pragma once
#include "Layer.h"
class GELULayer final : public Layer{
public:
	GELULayer(int batchSize, int channels, int height, int width, const char* layerName);
	~GELULayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetFineTune(bool enable) override;
	int batchSize_, outC_, outHeight_, outWidth_;
	__half* dataOut_ = nullptr;
	__half* data_ = nullptr;
};