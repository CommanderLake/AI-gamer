#include "Dropout.h"
#include "NNCommon.h"
#include "CuCommon.cuh"
#include <ctime>
Dropout::Dropout(const float dropoutRate, const int batchSize, const int channels, const int height, const int width, const std::string layerName, const bool train) : dropoutRate_(dropoutRate), keepProb_(1.0f - dropoutRate), batchSize_(batchSize), outC_(channels), outHeight_(height), outWidth_(width){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = outWidth_*outHeight_*outC_*batchSize_;
	if(dropoutRate_ > 0.0f && dropoutRate_ < 1.0f){
		checkCUDA(cudaMalloc(&mask_, outNCHW_*sizeof(unsigned char)));
		seed_ = static_cast<unsigned long long>(time(nullptr));
	}
}
Dropout::~Dropout(){
	cudaFree(mask_);
}
__half* Dropout::Forward(__half* data){
	if(!train_ || dropoutRate_ <= 0.0f){ return data; }
	if(dropoutRate_ >= 1.0f){
		checkCUDA(cudaMemset(data, 0, outNCHW_*sizeof(__half)));
		return data;
	}
	DropoutForward(data, mask_, static_cast<int>(outNCHW_), keepProb_, seed_++);
	return data;
}
__half* Dropout::Backward(__half* grad){
	if(!train_ || dropoutRate_ <= 0.0f){ return grad; }
	if(dropoutRate_ >= 1.0f){
		checkCUDA(cudaMemset(grad, 0, outNCHW_*sizeof(__half)));
		return grad;
	}
	DropoutBackward(grad, mask_, static_cast<int>(outNCHW_), keepProb_);
	return grad;
}
void Dropout::SetTrain(const bool enable){ train_ = enable; }