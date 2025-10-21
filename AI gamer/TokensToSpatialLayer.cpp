#include "TokensToSpatialLayer.h"
#include "CuCommon.cuh"
TokensToSpatialLayer::TokensToSpatialLayer(const int batchSize, const int nTokens, const int embedSize, const int patchRows, const int patchCols, const char* layerName, const bool train): ogbs_(batchSize), batchSize_(batchSize), nTokens_(nTokens), embedSize_(embedSize), patchRows_(patchRows), patchCols_(patchCols){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*nTokens_*embedSize_;
	CUDAMallocZero(&spatialData_, static_cast<size_t>(batchSize_)*embedSize_*nTokens_*sizeof(__half));
	CUDAMallocZero(&tokenGrad_, static_cast<size_t>(batchSize_)*nTokens_*embedSize_*sizeof(__half));
}
TokensToSpatialLayer::~TokensToSpatialLayer(){}
__half* TokensToSpatialLayer::Forward(__half* data){
	TokensToSpatial(data, spatialData_, batchSize_, nTokens_, embedSize_, patchRows_, patchCols_);
	return spatialData_;
}
__half* TokensToSpatialLayer::Backward(__half* grad){
	SpatialToTokens(grad, tokenGrad_, batchSize_, nTokens_, embedSize_, patchRows_, patchCols_);
	return tokenGrad_;
}
void TokensToSpatialLayer::SetFineTune(const bool enable){
	train_ = enable;
	batchSize_ = enable ? ogbs_ : 1;
}