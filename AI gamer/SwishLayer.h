#pragma once
#include "Layer.h"
class SwishLayer final : public Layer{
public:
	explicit SwishLayer(int batchSize, int channels, int height, int width, const char* layerName);
	~SwishLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	int batchSize_, outC_, outHeight_, outWidth_;
	__half* dataOut_ = nullptr;
	__half* data_ = nullptr;
};