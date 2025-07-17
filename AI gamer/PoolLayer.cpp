#include "PoolLayer.h"
#include "common.h"
#include "CuCommon.cuh"
PoolLayer::PoolLayer(const cudnnHandle_t cudnnHandle, const cudnnPoolingMode_t mode, const int batchSize, const int channels, int* height, int* width, const int poolSize, const int stride, const char* layerName, const bool train):
	cudnnHandle_(cudnnHandle), inHeight_(*height), inWidth_(*width), batchSize_(batchSize), outC_(channels){
	layerName_ = layerName;
	train_ = train;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, inHeight_, inWidth_));
	checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc_, mode, CUDNN_NOT_PROPAGATE_NAN, poolSize, poolSize, 0, 0, stride, stride));
	int n, c;
	checkCUDNN(cudnnGetPooling2dForwardOutputDim(poolDesc_, inDesc_, &n, &c, &outHeight_, &outWidth_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, outHeight_, outWidth_));
	outNCHW_ = batchSize_*outC_*outHeight_*outWidth_;
	inNCHW_ = batchSize_*outC_*inHeight_*inWidth_;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	if(train_){ CUDAMallocZero(&outGrad_, inNCHW_*sizeof(__half)); }
	*width = outWidth_;
	*height = outHeight_;
}
PoolLayer::~PoolLayer(){
	cudaFree(outData_);
	if(train_){
		cudaFree(outGrad_);
	}
}
__half* PoolLayer::Forward(__half* data){
	inData_ = data;
	checkCUDNN(cudnnPoolingForward(cudnnHandle_, poolDesc_, &alpha_, inDesc_, data, &beta0_, outDesc_, outData_));
	return outData_;
}
__half* PoolLayer::Backward(__half* grad){
	checkCUDNN(cudnnPoolingBackward(cudnnHandle_, poolDesc_, &alpha_, outDesc_, outData_, outDesc_, grad, inDesc_, inData_, &beta0_, inDesc_, outGrad_));
	return outGrad_;
}
void PoolLayer::SetTrain(const bool enable){
	int bs;
	if(enable){
		train_ = true;
		bs = batchSize_;
	} else{
		train_ = false;
		bs = 1;
	}
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, outC_, inHeight_, inWidth_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, outC_, outHeight_, outWidth_));
}