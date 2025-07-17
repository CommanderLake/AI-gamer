#include "SkipLayer.h"
#include "common.h"
SkipLayer::SkipLayer(cudnnHandle_t cudnnHandle, __half* inB, int batchSize, int channels, int height, int width, const char* layerName): cudnnHandle_(cudnnHandle), inB_(inB){
	layerName_ = layerName;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateOpTensorDescriptor(&opDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize, channels, height, width));
	checkCUDNN(cudnnSetOpTensorDescriptor(opDesc_, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_HALF, CUDNN_NOT_PROPAGATE_NAN));
}
SkipLayer::~SkipLayer(){
	cudnnDestroyOpTensorDescriptor(opDesc_);
	cudnnDestroyTensorDescriptor(inDesc_);
}
__half* SkipLayer::Forward(__half* data){
	checkCUDNN(cudnnOpTensor(cudnnHandle_, opDesc_, &alpha, inDesc_, data, &alpha, inDesc_, inB_, &beta, inDesc_, outData_));
	return outData_;
}
__half* SkipLayer::Backward(__half* grad){
	//TODO
}