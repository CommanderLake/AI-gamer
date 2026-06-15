#pragma once
#include "Layer.h"
class __declspec(dllexport) Dropout final : public Layer{
public:
	Dropout(float dropoutRate, int batchSize, int channels, int height, int width, std::string layerName, bool train);
	~Dropout() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	__half* ReplayForward(__half* data);
	void SetTrain(bool enable) override;
	float dropoutRate_;
	float keepProb_;
	int batchSize_, outC_, outHeight_, outWidth_;
	unsigned char* mask_ = nullptr;
	unsigned long long seed_ = 0;
};
