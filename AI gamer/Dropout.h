#pragma once
#include "Layer.h"

class Dropout : public Layer{
public:
	Dropout(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, float dropoutRate, const char* layerName);
	~Dropout() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	cudnnHandle_t cudnnHandle_;
	cudnnDropoutDescriptor_t dropoutDesc_;
	void* dropoutStates_;
	size_t stateSize_;
	void* reserveSpace_;
	size_t reserveSpaceSize_;
	float dropoutRate_;
};