#include "SpatialToTokensLayer.h"
#include "CuCommon.h"
SpatialToTokensLayer::SpatialToTokensLayer(const int batchSize, const int nTokens, const int embedSize, const int patchRows, const int patchCols, std::string layerName, const bool train) : batchSize_(batchSize), nTokens_(nTokens), embedSize_(embedSize), patchRows_(patchRows), patchCols_(patchCols){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = static_cast<size_t>(batchSize_) * nTokens_ * embedSize_;
	CUDAMallocZero(&tokenData_, outNCHW_ * sizeof(__half));
	CUDAMallocZero(&spatialGrad_, static_cast<size_t>(batchSize_) * embedSize_ * nTokens_ * sizeof(__half));
}
SpatialToTokensLayer::~SpatialToTokensLayer(){
	cudaFree(tokenData_);
	cudaFree(spatialGrad_);
}
__half* SpatialToTokensLayer::Forward(__half* data){
	SpatialToTokens(data, tokenData_, batchSize_, nTokens_, embedSize_, patchRows_, patchCols_);
	return tokenData_;
}
__half* SpatialToTokensLayer::Backward(__half* grad){
	TokensToSpatial(grad, spatialGrad_, batchSize_, nTokens_, embedSize_, patchRows_, patchCols_);
	return spatialGrad_;
}
void SpatialToTokensLayer::SetTrain(const bool enable){
	train_ = enable;
}
