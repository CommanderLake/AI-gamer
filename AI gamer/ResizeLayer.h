#pragma once
#include "Layer.h"

class ResizeLayer final : public Layer{
public:
	ResizeLayer(int batchSize, int channels, int inHeight, int inWidth, int outHeight, int outWidth, const char* layerName, bool train);
	~ResizeLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;

private:
	int batchSize_;
	int channels_;
	int inHeight_;
	int inWidth_;
	int outHeight_;
	int outWidth_;
	__half* outData_ = nullptr;
	__half* inGrad_ = nullptr;
};
