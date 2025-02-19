#pragma once
#include "Layer.h"
class SigmoidLayer final : public Layer {
public:
	SigmoidLayer(int numSigmoidOutputs, int batchSize, int outC, const char* layerName);
    ~SigmoidLayer() override;
    __half* Forward(__half* data) override;
    __half* Backward(__half* grad) override;
	size_t numSigmoidOutputs_;
	int outC_;
	int batchSize_;
	__half* data_;
};