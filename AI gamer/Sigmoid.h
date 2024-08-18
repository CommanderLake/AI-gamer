#pragma once
#include "Layer.h"
#include <cudnn.h>
class Sigmoid final : public Layer {
public:
    Sigmoid(cudnnTensorDescriptor_t outDesc, int numSigmoidOutputs, int batchSize, const char* layerName);
    ~Sigmoid() override;
    __half* Forward(__half* data) override;
    __half* Backward(__half* grad) override;
	size_t numSigmoidOutputs_;
	int outCHW_;
	int batchSize_;
	__half* data_;
};