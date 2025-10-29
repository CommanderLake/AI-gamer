#pragma once
#include "Layer.h"
#include <cuda_fp16.h>
class DiscardTokensLayer : public Layer{
public:
	DiscardTokensLayer(int batchSize, int totalTokens, int keepTokens, int embedDim, const char* layerName);
	~DiscardTokensLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	int batchSize_;
	int totalTokens_;
	int keepTokens_;
	int embedDim_;
	size_t keepCount_;
	size_t totalCount_;
	__half* outData_ = nullptr;
	__half* inGrad_ = nullptr;
};