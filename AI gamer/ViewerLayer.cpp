#include "ViewerLayer.h"
ViewerLayer::ViewerLayer(const int channels, const int height, const int width, const int gridWidth, const char* windowTitle, bool backwardPass, __half* displayData) : displayData_(displayData), backwardPass_(backwardPass), inC_(channels), inH_(height),
	inW_(width), gridW_(gridWidth){
	viewer_ = new Viewer();
	gridH_ = (inC_+gridW_-1)/gridW_;
	mosaicDimW_ = inW_*gridW_;
	mosaicDimH_ = inH_*gridH_;
	viewer_->InitializeWindow(mosaicDimW_, mosaicDimH_, windowTitle);
	mosaicH_ = static_cast<unsigned char*>(_mm_malloc(mosaicDimW_*mosaicDimH_, 64));
	CUDAMallocZero(&mosaicD_, mosaicDimW_*mosaicDimH_);
}
ViewerLayer::~ViewerLayer(){
	cudaFree(mosaicD_);
	_mm_free(mosaicH_);
	delete viewer_;
}
void ViewerLayer::DisplayData(const __half* data){
	FeatureMapMosaic(displayData_ == nullptr ? data : displayData_, mosaicD_, inH_, inW_, inC_, mosaicDimW_, inW_, inH_, gridW_);
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