#pragma once
#include <cudnn.h>
#include "Layer.h"
class Dropout final : public Layer{
public:
	Dropout(cudnnHandle_t cudnnHandle, float dropoutRate, int batchSize, int channels, int height, int width, std::string layerName, bool train);
	~Dropout() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	cudnnHandle_t cudnnHandle_;
	cudnnDropoutDescriptor_t dropoutDesc_;
	void* dropoutStates_;
	size_t stateSize_;
	void* reserveSpace_;
	size_t reserveSpaceSize_;
	float dropoutRate_;
	int batchSize_, outC_, outHeight_, outWidth_;
};