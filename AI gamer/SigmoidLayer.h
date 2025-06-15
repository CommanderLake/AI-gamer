#pragma once
#include "Layer.h"
class SigmoidLayer final : public Layer {
public:
	SigmoidLayer(int batchSize, int numCtrls, int numButs, const char* layerName);
    ~SigmoidLayer() override;
    __half* Forward(__half* data) override;
    __half* Backward(__half* grad) override;
	int batchSize_, numCtrls_, numButs_;
	__half* data_ = nullptr;
};