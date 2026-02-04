#include "ResizeLayer.h"
#include "CuCommon.cuh"

ResizeLayer::ResizeLayer(const int batchSize, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth, std::string layerName, const bool train)
	: batchSize_(batchSize),
	  channels_(channels),
	  inHeight_(inHeight),
	  inWidth_(inWidth),
	  outHeight_(outHeight),
	  outWidth_(outWidth){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = static_cast<size_t>(batchSize_)*channels_*outHeight_*outWidth_;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&inGrad_, static_cast<size_t>(batchSize_)*channels_*inHeight_*inWidth_*sizeof(__half));
}

ResizeLayer::~ResizeLayer(){
	cudaFree(outData_);
	cudaFree(inGrad_);
}

__half* ResizeLayer::Forward(__half* data){
	ScaleNearestNeighborForward(data, outData_, batchSize_, channels_, inHeight_, inWidth_, outHeight_, outWidth_);
	return outData_;
}

__half* ResizeLayer::Backward(__half* grad){
	checkCUDA(cudaMemset(inGrad_, 0, static_cast<size_t>(batchSize_)*channels_*inHeight_*inWidth_*sizeof(__half)));
	ScaleNearestNeighborBackward(grad, inGrad_, batchSize_, channels_, inHeight_, inWidth_, outHeight_, outWidth_);
	return inGrad_;
}

void ResizeLayer::SetTrain(const bool enable){
	train_ = enable;
}
