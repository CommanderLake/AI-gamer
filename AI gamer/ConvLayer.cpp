#include "ConvLayer.h"
#include "common.h"
#include <iostream>
#include <algorithm>
ConvLayer::ConvLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int bitchSize, int inputChannels, int outputChannels, int filterSize, int stride, int padding, int* width, int* height, int* inputCHW, int outW,
					int outH, const char* layerName) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), fwdAlgo_(), bwdFilterAlgo_(), bwdDataAlgo_(), bitchSize_(bitchSize), inC_(inputChannels), inCHW_(*inputCHW),
												outC_(outputChannels), inData_(nullptr), outData_(nullptr), weights_(nullptr){
	layerName_ = layerName;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outGradDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&inGradDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc_));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(biasDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, outC_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bitchSize_, inC_, *height, *width));
	checkCUDNN(cudnnSetTensor4dDescriptor(outGradDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bitchSize_, inC_, *height, *width));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, outC_, inC_, filterSize, filterSize));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc_, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolutionMathType(convDesc_, CUDNN_TENSOR_OP_MATH));//S
	int n, c;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc_, inDesc_, filterDesc_, &n, &c, height, width));
	if(outW > 0) *width = outW;
	if(outH > 0) *height = outH;
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, *height, *width));
	checkCUDNN(cudnnSetTensor4dDescriptor(inGradDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, *height, *width));
	outCHW_ = outC_ * *height * *width;
	outNCHW_ = outCHW_*bitchSize_;
	gradOutSize_ = inCHW_*bitchSize_*sizeof(__half);
	const auto fanIn = inC_*filterSize*filterSize;
	weightCount_ = outC_ * fanIn;
	checkCUDA(cudaMalloc(&outData_, outNCHW_*sizeof(__half)));
	checkCUDA(cudaMalloc(&weights_, weightCount_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradWeights_, weightCount_*sizeof(__half)));
	checkCUDA(cudaMalloc(&bias_, outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradBias_, outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradOut_, gradOutSize_));
	checkCUDA(cudaMalloc(&m_Weights_, weightCount_*sizeof(float)));
	checkCUDA(cudaMalloc(&v_Weights_, weightCount_*sizeof(float)));
	checkCUDA(cudaMalloc(&m_Bias_, outC_*sizeof(float)));
	checkCUDA(cudaMalloc(&v_Bias_, outC_*sizeof(float)));
	checkCUDA(cudaMemset(outData_, 0, outNCHW_*sizeof(__half)));
	HeInit(weights_, weightCount_, fanIn);
	checkCUDA(cudaMemset(gradWeights_, 0, weightCount_*sizeof(__half)));
	checkCUDA(cudaMemset(bias_, 0, outC_*sizeof(__half)));
	checkCUDA(cudaMemset(gradBias_, 0, outC_*sizeof(__half)));
	checkCUDA(cudaMemset(gradOut_, 0, gradOutSize_));
	checkCUDA(cudaMemset(m_Weights_, 0, weightCount_*sizeof(float)));
	checkCUDA(cudaMemset(v_Weights_, 0, weightCount_*sizeof(float)));
	checkCUDA(cudaMemset(m_Bias_, 0, outC_*sizeof(float)));
	checkCUDA(cudaMemset(v_Bias_, 0, outC_*sizeof(float)));
	cudnnConvolutionFwdAlgoPerf_t fwdAlgoPerf[10];
	cudnnConvolutionBwdFilterAlgoPerf_t bwdFilterAlgoPerf[10];
	cudnnConvolutionBwdDataAlgoPerf_t bwdDataAlgoPerf[10];
	int returnedAlgoCount;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_, inDesc_, filterDesc_, convDesc_, outDesc_, 10, &returnedAlgoCount, fwdAlgoPerf));
	fwdAlgo_ = fwdAlgoPerf[0].algo;
	const size_t fwdWorkspaceSize = fwdAlgoPerf[0].memory;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_, inDesc_, outDesc_, convDesc_, filterDesc_, 10, &returnedAlgoCount, bwdFilterAlgoPerf));
	bwdFilterAlgo_ = bwdFilterAlgoPerf[0].algo;
	const size_t bwdFilterWorkspaceSize = bwdFilterAlgoPerf[0].memory;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_, filterDesc_, outDesc_, convDesc_, inDesc_, 10, &returnedAlgoCount, bwdDataAlgoPerf));
	bwdDataAlgo_ = bwdDataAlgoPerf[0].algo;
	const size_t bwdDataWorkspaceSize = bwdDataAlgoPerf[0].memory;
	workspaceSize_ = std::max(fwdWorkspaceSize, std::max(bwdFilterWorkspaceSize, bwdDataWorkspaceSize));
	checkCUDA(cudaMalloc(&workspace_, workspaceSize_));
	*inputCHW = outCHW_;
}
ConvLayer::~ConvLayer(){
	cudaFree(outData_);
	cudaFree(bias_);
	cudaFree(weights_);
	cudaFree(gradOut_);
	cudaFree(gradWeights_);
	cudaFree(gradBias_);
	cudaFree(workspace_);
	cudaFree(m_Weights_);
	cudaFree(v_Weights_);
	cudaFree(m_Bias_);
	cudaFree(v_Bias_);
	checkCUDNN(cudnnDestroyTensorDescriptor(inDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc_));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasDesc_));
}
__half* ConvLayer::Forward(__half* data){
	inData_ = data;
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, inDesc_, data, filterDesc_, weights_, convDesc_, fwdAlgo_, workspace_, workspaceSize_, &beta0, outDesc_, outData_));
	cudnnAddTensor(cudnnHandle_, &alpha, biasDesc_, bias_, &beta1, outDesc_, outData_);
	return outData_;
}
__half* ConvLayer::Backward(__half* grad){
	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &alpha, inDesc_, inData_, inGradDesc_, grad, convDesc_, bwdFilterAlgo_, workspace_, workspaceSize_, &beta0, filterDesc_, gradWeights_));
	checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle_, &alpha, inGradDesc_, grad, &beta0, biasDesc_, gradBias_));
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &alpha, filterDesc_, weights_, inGradDesc_, grad, convDesc_, bwdDataAlgo_, workspace_, workspaceSize_, &beta0, outGradDesc_, gradOut_));
	return gradOut_;
}
void ConvLayer::UpdateParameters(float learningRate){
	//SGDHalf(weights_, learningRate, gradWeights_, weightCount_);
	//SGDHalf(bias_, learningRate, gradBias_, outC_);
	AdamWHalf(weights_, m_Weights_, v_Weights_, learningRate, gradWeights_, weightCount_, t_, 0.001F);
	AdamWHalf(bias_, m_Bias_, v_Bias_, learningRate, gradBias_, outC_, t_, 0.001F);
	++t_;
}