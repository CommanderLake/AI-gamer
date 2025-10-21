#pragma once
#include "Layer.h"
class TokensToSpatialLayer : public Layer{
public:
	TokensToSpatialLayer::TokensToSpatialLayer(int batchSize, int nTokens, int embedSize, int patchRows, int patchCols, const char* layerName, bool train);
	TokensToSpatialLayer::~TokensToSpatialLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetFineTune(bool enable) override;
	int ogbs_, batchSize_, nTokens_, embedSize_;
	int patchRows_, patchCols_;
	__half* spatialData_ = nullptr;
	__half* tokenGrad_ = nullptr;
};