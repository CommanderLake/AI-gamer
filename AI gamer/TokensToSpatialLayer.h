#pragma once
#include "Layer.h"
class TokensToSpatialLayer : public Layer{
public:
	TokensToSpatialLayer::TokensToSpatialLayer(int batchSize, int nTokens, int embedSize, int patchRows, int patchCols, std::string layerName, bool train);
	TokensToSpatialLayer::~TokensToSpatialLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	int batchSize_, nTokens_, embedSize_;
	int patchRows_, patchCols_;
	__half* spatialData_ = nullptr;
	__half* tokenGrad_ = nullptr;
};