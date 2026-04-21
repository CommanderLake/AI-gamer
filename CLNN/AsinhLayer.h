#pragma once
#include "Layer.h"
class __declspec(dllexport) AsinhLayer final : public Layer{
public:
	AsinhLayer(int batchSize, int channels, int height, int width, float alpha, std::string layerName);
	~AsinhLayer() override = default;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
private:
	int batchSize_;
	int outC_;
	int outHeight_;
	int outWidth_;
	float alpha_;
	__half* data_ = nullptr;
};