#include "Dropout.h"
#include "common.h"
Dropout::Dropout(cudnnHandle_t cudnnHandle, float dropoutRate, int batchSize, int channels, int height, int width, const char* layerName, bool train) : cudnnHandle_(cudnnHandle), dropoutRate_(dropoutRate){
	layerName_ = layerName;
	train_ = train;
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
	if(train_){
		checkCUDNN(cudnnDropoutForward(cudnnHandle_, dropoutDesc_, outDesc_, data, outDesc_, data, reserveSpace_, reserveSpaceSize_));
	}
	return data;
}
__half* Dropout::Backward(__half* grad){
	checkCUDNN(cudnnDropoutBackward(cudnnHandle_, dropoutDesc_, outDesc_, grad, outDesc_, grad, reserveSpace_, reserveSpaceSize_));
	return grad;
}