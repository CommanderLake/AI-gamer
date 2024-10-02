#include "Activate.h"
#include "common.h"
Activate::Activate(cudnnHandle_t cudnnHandle, cudnnActivationMode_t mode, double coef, int batchSize, int channels, int height, int width, const char* layerName): cudnnHandle_(cudnnHandle){
	layerName_ = layerName;
	outNCHW_ = batchSize*channels*height*width;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, channels, height, width));
	checkCUDNN(cudnnCreateTensorDescriptor(&gradDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(gradDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, channels, height, width));
	checkCUDNN(cudnnCreateActivationDescriptor(&activDesc_));
	checkCUDNN(cudnnSetActivationDescriptor(activDesc_, mode, CUDNN_NOT_PROPAGATE_NAN, coef));
}
Activate::~Activate(){
	cudnnDestroyTensorDescriptor(gradDesc_);
	cudnnDestroyActivationDescriptor(activDesc_);
}
__half* Activate::Forward(__half* data){
	data_ = data;
	checkCUDNN(cudnnActivationForward(cudnnHandle_, activDesc_, &alpha, outDesc_, data, &beta0, outDesc_, data));
	return data;
}
__half* Activate::Backward(__half* grad){
	checkCUDNN(cudnnActivationBackward(cudnnHandle_, activDesc_, &alpha, outDesc_, data_, gradDesc_, grad, outDesc_, data_, &beta1, gradDesc_, grad));
	return grad;
}