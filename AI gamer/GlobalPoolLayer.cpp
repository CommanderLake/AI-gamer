#include "GlobalPoolLayer.h"
#include "CuCommon.cuh"
GlobalPoolLayer::GlobalPoolLayer(int batchSize, int tokens, int embedDim, const char* layerName, bool train) : ogbs_(batchSize), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*embedDim_;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	if(train_){ CUDAMallocZero(&outGrad_, batchSize_*tokens_*embedDim_*sizeof(__half)); }
}
GlobalPoolLayer::~GlobalPoolLayer(){
	cudaFree(outData_);
	if(train_) cudaFree(outGrad_);
}
__half* GlobalPoolLayer::Forward(__half* data){
	GlobalAvgPoolForward(data, outData_, batchSize_, tokens_, embedDim_);
	return outData_;
}
__half* GlobalPoolLayer::Backward(__half* grad){
	GlobalAvgPoolBackward(grad, outGrad_, batchSize_, tokens_, embedDim_);
	return outGrad_;
}
void GlobalPoolLayer::SetTrain(bool enable){
	train_ = enable;
	batchSize_ = enable ? ogbs_ : 1;
	outNCHW_ = batchSize_*embedDim_;
}