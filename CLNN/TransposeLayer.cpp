#include "TransposeLayer.h"
#include "NNCommon.h"
#include "CuCommon.cuh"
TransposeLayer::TransposeLayer(const cudnnHandle_t cudnnHandle, const int batchSize, const int channels, const int height, const int width, const cudnnTensorFormat_t inFormat, const cudnnTensorFormat_t outFormat, std::string layerName): cudnnHandle_(cudnnHandle){
	layerName_ = layerName;
	outNCHW_ = batchSize*channels*height*width;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, inFormat, CUDNN_DATA_HALF, batchSize, channels, height, width));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, outFormat, CUDNN_DATA_HALF, batchSize, channels, height, width));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&outGrad_, outNCHW_*sizeof(__half));
}
TransposeLayer::~TransposeLayer(){
	cudnnDestroyTensorDescriptor(inDesc_);
	cudnnDestroyTensorDescriptor(outDesc_);
	cudaFree(outData_);
	cudaFree(outGrad_);
}
__half* TransposeLayer::Forward(__half* data){ 
	checkCUDNN(cudnnTransformTensor(cudnnHandle_, &alpha_, inDesc_, data, &beta_, outDesc_, outData_));
	return outData_;
}
__half* TransposeLayer::Backward(__half* grad){
	checkCUDNN(cudnnTransformTensor(cudnnHandle_, &alpha_, outDesc_, grad, &beta_, inDesc_, outGrad_));
	return outGrad_;
}