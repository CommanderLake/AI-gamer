#include "Dropout.h"
#include "NNCommon.h"
#include "CuCommon.cuh"
#include <ctime>
Dropout::Dropout(const cudnnHandle_t cudnnHandle, const float dropoutRate, const int batchSize, const int channels, const int height, const int width, const std::string layerName, const bool train) : cudnnHandle_(cudnnHandle), dropoutRate_(dropoutRate), batchSize_(batchSize), outC_(channels), outHeight_(height), outWidth_(width){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = outWidth_*outHeight_*outC_*batchSize_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, channels, height, width));
	checkCUDNN(cudnnDropoutGetStatesSize(cudnnHandle_, &stateSize_));
	checkCUDA(cudaMalloc(&dropoutStates_, stateSize_));
	checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc_));
	checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc_, cudnnHandle_, dropoutRate_, dropoutStates_, stateSize_, static_cast<unsigned long long>(time(nullptr))));
	checkCUDNN(cudnnDropoutGetReserveSpaceSize(outDesc_, &reserveSpaceSize_));
	checkCUDA(cudaMalloc(&reserveSpace_, reserveSpaceSize_));
}
Dropout::~Dropout(){
	checkCUDNN(cudnnDestroyDropoutDescriptor(dropoutDesc_));
	checkCUDA(cudaFree(dropoutStates_));
	checkCUDA(cudaFree(reserveSpace_));
}
__half* Dropout::Forward(__half* data){
	if(train_){ checkCUDNN(cudnnDropoutForward(cudnnHandle_, dropoutDesc_, outDesc_, data, outDesc_, data, reserveSpace_, reserveSpaceSize_)); }
	return data;
}
__half* Dropout::Backward(__half* grad){
	checkCUDNN(cudnnDropoutBackward(cudnnHandle_, dropoutDesc_, outDesc_, grad, outDesc_, grad, reserveSpace_, reserveSpaceSize_));
	return grad;
}
void Dropout::SetTrain(const bool enable){ train_ = enable; }