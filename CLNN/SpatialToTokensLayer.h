#pragma once
#include "Layer.h"
class __declspec(dllexport) SpatialToTokensLayer : public Layer{
public:
	SpatialToTokensLayer(int batchSize, int nTokens, int embedSize, int patchRows, int patchCols, std::string layerName, bool train);
	~SpatialToTokensLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTrain(bool enable) override;
	int batchSize_, nTokens_, embedSize_;
	int patchRows_, patchCols_;
	__half* tokenData_ = nullptr;
	__half* spatialGrad_ = nullptr;
};
