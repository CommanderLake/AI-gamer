#include "DiscardTokensLayer.h"
#include "CuCommon.cuh"
DiscardTokensLayer::DiscardTokensLayer(const int batchSize, const int totalTokens, const int keepTokens, const int embedDim, const char* layerName) : batchSize_(batchSize), totalTokens_(totalTokens), keepTokens_(keepTokens), embedDim_(embedDim){
	layerName_ = layerName;
	train_ = false;
	keepCount_ = static_cast<size_t>(batchSize_) * keepTokens_ * embedDim_;
	totalCount_ = static_cast<size_t>(batchSize_) * totalTokens_ * embedDim_;
	outNCHW_ = keepCount_;
	if(keepCount_ > 0){ CUDAMallocZero(&outData_, keepCount_ * sizeof(__half)); }
	if(totalCount_ > 0){ CUDAMallocZero(&inGrad_, totalCount_ * sizeof(__half)); }
}
DiscardTokensLayer::~DiscardTokensLayer(){
	if(outData_){ cudaFree(outData_); }
	if(inGrad_){ cudaFree(inGrad_); }
}
__half* DiscardTokensLayer::Forward(__half* data){
	if(keepCount_ > 0){ checkCUDA(cudaMemcpy(outData_, data, keepCount_*sizeof(__half), cudaMemcpyDeviceToDevice)); }
	return outData_;
}
__half* DiscardTokensLayer::Backward(__half* grad){
	if(totalCount_ > 0){
		checkCUDA(cudaMemset(inGrad_, 0, totalCount_*sizeof(__half)));
		if(keepCount_ > 0){ checkCUDA(cudaMemcpy(inGrad_, grad, keepCount_*sizeof(__half), cudaMemcpyDeviceToDevice)); }
	}
	return inGrad_;
}
void DiscardTokensLayer::SetTrain(const bool enable){ train_ = enable; }