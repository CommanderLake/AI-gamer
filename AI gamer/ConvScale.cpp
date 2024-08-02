#include "ConvScale.h"
#include "common.h"
ConvScale::ConvScale(cudnnHandle_t cudnnHandle, const int scaleFactor, int inWidth, int inHeight) : cudnnHandle_(cudnnHandle), dFilter_(nullptr), dWorkspace_(nullptr), inWidth_(inWidth), inHeight_(inHeight), scaleFactor_(scaleFactor){
	inHWC_ = inWidth_*inHeight_*3;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc_));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, 3, inHeight_, inWidth_));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, 3, 3, scaleFactor_, scaleFactor_));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc_, 0, 0, scaleFactor_, scaleFactor_, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolutionMathType(convDesc_, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)); //S
	int n, c;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc_, inDesc_, filterDesc_, &n, &c, &outHeight_, &outWidth_));
	outHWC_ = outWidth_*outHeight_*3;
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, outHeight_, outWidth_));
	const int filterSize = scaleFactor_*scaleFactor_;
	const std::vector<__half> hFilter(9*filterSize, __float2half(1.0f/filterSize));
	checkCUDA(cudaMalloc(&dFilter_, hFilter.size()*sizeof(__half)));
	checkCUDA(cudaMemcpy(dFilter_, hFilter.data(), hFilter.size()*sizeof(__half), cudaMemcpyHostToDevice));
	checkCUDA(cudaMalloc(&dIn_, inHWC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&dOut_, outHWC_*sizeof(__half)));
	checkCUDA(cudaMemset(dIn_, 0, inHWC_*sizeof(__half)));
	checkCUDA(cudaMemset(dOut_, 0, outHWC_*sizeof(__half)));
	cudnnConvolutionFwdAlgoPerf_t algoPerf;
	int returnedAlgoCount;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_, inDesc_, filterDesc_, convDesc_, outDesc_, 1, &returnedAlgoCount, &algoPerf));
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_, inDesc_, filterDesc_, convDesc_, outDesc_, algoPerf.algo, &workspaceBytes_));
	checkCUDA(cudaMalloc(&dWorkspace_, workspaceBytes_));
}
ConvScale::~ConvScale(){
	cudaFree(dFilter_);
	cudaFree(dWorkspace_);
	cudaFree(dIn_);
	cudaFree(dOut_);
	cudnnDestroyTensorDescriptor(inDesc_);
	cudnnDestroyTensorDescriptor(outDesc_);
	cudnnDestroyFilterDescriptor(filterDesc_);
	cudnnDestroyConvolutionDescriptor(convDesc_);
}
void ConvScale::ScaleInPlace(unsigned char* inImage){
	ConvertAndNormalize(dIn_, inImage, inHWC_);
	constexpr float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, inDesc_, dIn_, filterDesc_, dFilter_, convDesc_, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, dWorkspace_, workspaceBytes_, &beta, outDesc_, dOut_));
	UnConvertAndUnNormalize(inImage, dOut_, outHWC_);
}
__half* ConvScale::ScaleToHalf(unsigned char* inImage){
	ConvertAndNormalize(dIn_, inImage, inHWC_);
	constexpr float alpha = 1.0f, beta = 0.0f;
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, inDesc_, dIn_, filterDesc_, dFilter_, convDesc_, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, dWorkspace_, workspaceBytes_, &beta, outDesc_, dOut_));
	return dOut_;
}