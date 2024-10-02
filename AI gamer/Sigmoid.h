#pragma once
#include "Layer.h"
class Sigmoid final : public Layer {
public:
	Sigmoid(int numSigmoidOutputs, int batchSize, int outC, const char* layerName);
    ~Sigmoid() override;
    __half* Forward(__half* data) override;
    __half* Backward(__half* grad) override;
	size_t numSigmoidOutputs_;
	int outC_;
	int batchSize_;
	__half* data_;
};