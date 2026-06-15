#include "ResizeLayer.h"
#include "CuCommon.h"
#include <algorithm>
#include <stdexcept>
ResizeLayer::ResizeLayer(const int batchSize, const int channels, const int inHeight, const int inWidth, const int outHeight, const int outWidth, std::string layerName, const bool train) : batchSize_(batchSize), channels_(channels), inHeight_(inHeight), inWidth_(inWidth), outHeight_(outHeight), outWidth_(outWidth){
	layerName_ = layerName;
	train_ = train;
	if(batchSize_ <= 0 || channels_ <= 0 || inHeight_ <= 0 || inWidth_ <= 0 || outHeight_ <= 0 || outWidth_ <= 0){ throw std::invalid_argument("ResizeLayer dimensions must be positive"); }
	const long long widthLimitedHeight = (static_cast<long long>(inHeight_)*outWidth_ + inWidth_/2)/inWidth_;
	if(widthLimitedHeight <= outHeight_){
		contentWidth_ = outWidth_;
		contentHeight_ = std::max(1, static_cast<int>(widthLimitedHeight));
	} else{
		contentHeight_ = outHeight_;
		contentWidth_ = std::max(1, static_cast<int>((static_cast<long long>(inWidth_)*outHeight_ + inHeight_/2)/inHeight_));
	}
	contentTop_ = (outHeight_ - contentHeight_)/2;
	contentLeft_ = (outWidth_ - contentWidth_)/2;
	outNCHW_ = static_cast<size_t>(batchSize_) * channels_ * outHeight_ * outWidth_;
	CUDAMallocZero(&outData_, outNCHW_ * sizeof(__half));
	CUDAMallocZero(&inGrad_, static_cast<size_t>(batchSize_) * channels_ * inHeight_ * inWidth_ * sizeof(__half));
}
ResizeLayer::~ResizeLayer(){
	cudaFree(outData_);
	cudaFree(inGrad_);
}
__half* ResizeLayer::Forward(__half* data){
	ScaleNearestNeighborLetterboxForward(data, outData_, batchSize_, channels_, inHeight_, inWidth_, outHeight_, outWidth_, contentTop_, contentLeft_, contentHeight_, contentWidth_);
	return outData_;
}
__half* ResizeLayer::Backward(__half* grad){
	checkCUDA(cudaMemset(inGrad_, 0, static_cast<size_t>(batchSize_)*channels_*inHeight_*inWidth_*sizeof(__half)));
	ScaleNearestNeighborLetterboxBackward(grad, inGrad_, batchSize_, channels_, inHeight_, inWidth_, outHeight_, outWidth_, contentTop_, contentLeft_, contentHeight_, contentWidth_);
	return inGrad_;
}
void ResizeLayer::SetTrain(const bool enable){ train_ = enable; }
