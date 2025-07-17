#include "ConvScale.h"
#include "common.h"
#include "CuCommon.cuh"
ConvScale::ConvScale(const cudnnHandle_t cudnnHandle, const int filterSize, const int stride, const int padding, const int batchSize, const int channels, int* height, int* width) :
cudnnHandle_(cudnnHandle), dFilter_(nullptr), dWorkspace_(nullptr), batchSize_(batchSize), inC_(channels), inWidth_(*width), inHeight_(*height), stride_(stride), filterSize_(filterSize), padding_(padding){
	inNCHW_ = batchSize_*inC_*inHeight_*inWidth_;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc_));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, inHeight_, inWidth_));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, inC_, inC_, filterSize_, filterSize_));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc_, padding_, padding_, stride_, stride_, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolutionMathType(convDesc_, CUDNN_TENSOR_OP_MATH)); //S
	int n, c;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc_, inDesc_, filterDesc_, &n, &c, height, width));
	outWidth_ = *width;
	outHeight_ = *height;
	outNCHW_ = batchSize_*inC_**height**width;
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, *height, *width));
	size_t filterSizeB;
	checkCUDNN(cudnnGetFilterSizeInBytes(filterDesc_, &filterSizeB));
	const int filterSqu = filterSize_*filterSize_;
	const auto hFilter = static_cast<__half*>(_mm_malloc(filterSizeB, 64));
	for(int ch = 0; ch<inC_; ++ch){ for(int f = 0; f<inC_; ++f){ for(int i = 0; i<filterSqu; ++i){ hFilter[(ch*inC_+f)*filterSqu+i] = __float2half(ch==f ? 1.0f/filterSqu : 0.0f); } } }
	checkCUDA(cudaMalloc(&dFilter_, filterSizeB));
	checkCUDA(cudaMemcpy(dFilter_, hFilter, filterSizeB, cudaMemcpyHostToDevice));
	CUDAMallocZero(&dInB_, inNCHW_);
	CUDAMallocZero(&dIn_, inNCHW_*sizeof(__half));
	CUDAMallocZero(&dOut_, outNCHW_*sizeof(__half));
	int returnedAlgoCount;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_, inDesc_, filterDesc_, convDesc_, outDesc_, 1, &returnedAlgoCount, &algoPerf_));
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_, inDesc_, filterDesc_, convDesc_, outDesc_, algoPerf_.algo, &workspaceBytes_));
	checkCUDA(cudaMalloc(&dWorkspace_, workspaceBytes_));
	_mm_free(hFilter);
}
ConvScale::~ConvScale(){
	cudaFree(dFilter_);
	cudaFree(dWorkspace_);
	cudaFree(dIn_);
	cudaFree(dOut_);
	cudaFree(dInB_);
	cudnnDestroyTensorDescriptor(inDesc_);
	cudnnDestroyTensorDescriptor(outDesc_);
	cudnnDestroyFilterDescriptor(filterDesc_);
	cudnnDestroyConvolutionDescriptor(convDesc_);
}
void ConvScale::Forward(){
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha_, inDesc_, dIn_, filterDesc_, dFilter_, convDesc_, algoPerf_.algo, dWorkspace_, workspaceBytes_, &beta_, outDesc_, dOut_));
}
void ConvScale::ScaleUInt8InPlaceHost(unsigned char* inImage){
	cudaMemcpy(dInB_, inImage, inNCHW_, cudaMemcpyHostToDevice);
	ConvertByteToHalf(dInB_, dIn_, inNCHW_, false);
	Forward();
	ConvertHalfToByte(dOut_, dInB_, outNCHW_, false);
	cudaMemcpy(inImage, dInB_, outNCHW_, cudaMemcpyDeviceToHost);
}
void ConvScale::ScaleUInt8InPlaceDevice(unsigned char* inImage){
	ConvertByteToHalf(inImage, dIn_, inNCHW_, false);
	Forward();
	ConvertHalfToByte(dOut_, inImage, outNCHW_, false);
}
__half* ConvScale::ScaleUInt8ToFP16Device(const unsigned char* inImage){
	ConvertByteToHalf(inImage, dIn_, inNCHW_, true);
	Forward();
	return dOut_;
}