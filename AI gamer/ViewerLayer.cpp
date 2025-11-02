#include "ViewerLayer.h"
#include "CuCommon.cuh"
ViewerLayer::ViewerLayer(int dataSize, const int channels, const int patchHeight, const int patchWidth, const int gridWidth, const std::string windowTitle, bool mosaic, const float scale, bool backwardPass, __half* displayData) : displayData_(displayData), windowTitle_(windowTitle),
	backwardPass_(backwardPass), inC_(channels), inH_(patchHeight), inW_(patchWidth), gridW_(gridWidth), scale_(scale), mosaic_(mosaic){
	outNCHW_ = dataSize;
	layerName_ = windowTitle_.c_str();
	viewer_ = new Viewer();
	gridH_ = (inC_ + gridW_ - 1) / gridW_;
	mosaicDimW_ = inW_*gridW_;
	mosaicDimH_ = inH_*gridH_;
	viewer_->InitializeWindow(mosaicDimW_, mosaicDimH_, windowTitle_.c_str());
	mosaicH_ = static_cast<unsigned char*>(_mm_malloc(mosaicDimW_*mosaicDimH_, 64));
	CUDAMallocZero(&mosaicD_, mosaicDimW_*mosaicDimH_);
}
ViewerLayer::~ViewerLayer(){
	cudaFree(mosaicD_);
	_mm_free(mosaicH_);
	delete viewer_;
}
void ViewerLayer::DisplayData(const __half* data){
	if(mosaic_) FeatureMapMosaic(displayData_ == nullptr ? data : displayData_, mosaicD_, inH_, inW_, inC_, mosaicDimW_, inW_, inH_, gridW_, scale_);
	else ConvertHalfToByte(data, mosaicD_, mosaicDimW_*mosaicDimH_, true);
	checkCUDA(cudaMemcpy(mosaicH_, mosaicD_, mosaicDimW_*mosaicDimH_, cudaMemcpyDeviceToHost));
	viewer_->ShowImageGreyscale(mosaicH_, mosaicDimW_, mosaicDimH_);
}
__half* ViewerLayer::Forward(__half* data){
	if(!backwardPass_) DisplayData(data);
	return data;
}
__half* ViewerLayer::Backward(__half* grad){
	if(backwardPass_) DisplayData(grad);
	return grad;
}