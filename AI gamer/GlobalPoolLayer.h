#pragma once
#include "Layer.h"
#include <cuda_runtime.h>

class GlobalPoolLayer final : public Layer{
public:
	GlobalPoolLayer(int batchSize, int tokens, int embedDim, const char* layerName, bool train);
	~GlobalPoolLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;

private:
	int ogbs_, batchSize_, tokens_, embedDim_;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
};