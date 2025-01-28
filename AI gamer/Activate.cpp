#include "Activate.h"
#include "common.h"
Activate::Activate(cudnnHandle_t cudnnHandle, cudnnActivationMode_t mode, double coef, int batchSize, int channels, int height, int width, const char* layerName): cudnnHandle_(cudnnHandle), batchSize_(batchSize), outC_(channels), outHeight_(height), outWidth_(width){
	layerName_ = layerName;
	outNCHW_ = batchSize*channels*height*width;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, channels, height, width));
	checkCUDNN(cudnnCreateActivationDescriptor(&activDesc_));
	checkCUDNN(cudnnSetActivationDescriptor(activDesc_, mode, CUDNN_NOT_PROPAGATE_NAN, coef));
	CUDAMallocZero(&dataOut_, outNCHW_*sizeof(__half));
}
Activate::~Activate(){
	cudnnDestroyActivationDescriptor(activDesc_);
}
__half* Activate::Forward(__half* data){
	dataIn_ = data;
	checkCUDNN(cudnnActivationForward(cudnnHandle_, activDesc_, &alpha, outDesc_, dataIn_, &beta0, outDesc_, dataOut_));
	return dataOut_;
}
__half* Activate::Backward(__half* grad){
	checkCUDNN(cudnnActivationBackward(cudnnHandle_, activDesc_, &alpha, outDesc_, dataOut_, outDesc_, grad, outDesc_, dataIn_, &beta1, outDesc_, grad));
	return grad;
}
void Activate::SetTrain(bool enable){
	int bs;
	if(enable){
		train_ = true;
		bs = batchSize_;
	} else{
		train_ = false;
		bs = 1;
	}
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, outC_, outHeight_, outWidth_));
}