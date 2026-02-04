#pragma once
#include "Layer.h"
class LeakyReLU final : public Layer{
public:
	explicit LeakyReLU(int batchSize, int channels, int height, int width, std::string layerName);
	~LeakyReLU() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	int batchSize_, outC_, outHeight_, outWidth_;
	float slope_;
	__half* dataOut_ = nullptr;
	__half* data_ = nullptr;
};