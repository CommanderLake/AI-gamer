#include "Dropout.h"
#include "common.h"
Dropout::Dropout(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, float dropoutRate, const char* layerName) : cudnnHandle_(cudnnHandle), dropoutRate_(dropoutRate){
	layerName_ = layerName;
	outDesc_ = outDesc;
	checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc_));
	checkCUDNN(cudnnDropoutGetStatesSize(cudnnHandle_, &stateSize_));
	checkCUDA(cudaMalloc(&dropoutStates_, stateSize_));
	checkCUDNN(cudnnDropoutGetReserveSpaceSize(outDesc_, &reserveSpaceSize_));
	checkCUDA(cudaMalloc(&reserveSpace_, reserveSpaceSize_));
	checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc_, cudnnHandle_, dropoutRate_, dropoutStates_, stateSize_, static_cast<unsigned long long>(time(nullptr))));
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
	return grad;
}