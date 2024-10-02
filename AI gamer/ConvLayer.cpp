#include "ConvLayer.h"
#include <iostream>
#include <algorithm>
ConvLayer::ConvLayer(cudnnHandle_t cudnnHandle, int batchSize, int inputChannels, int outputChannels, int filterSize, int stride, int padding, int* height, int* width, const char* layerName, bool train, float weightDecay) : cudnnHandle_(
		cudnnHandle), batchSize_(batchSize), inC_(inputChannels), outC_(outputChannels), inData_(nullptr), outData_(nullptr), weights_(nullptr), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	int inWidth = *width, inHeight = *height, outWidth, outHeight;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc_));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(biasDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, outC_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, inHeight, inWidth));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, outC_, inC_, filterSize, filterSize));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc_, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolutionMathType(convDesc_, CUDNN_TENSOR_OP_MATH)); //S
	int n, c;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc_, inDesc_, filterDesc_, &n, &c, &outHeight, &outWidth));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, outHeight, outWidth));
	outNCHW_ = outWidth*outHeight*outC_*batchSize_;
	gradOutSize_ = inWidth*inHeight*inC_*batchSize_*sizeof(__half);
	const auto fanIn = inC_*filterSize*filterSize;
	//const auto fanOut = outC_*filterSize*filterSize;
	weightCount_ = outC_*fanIn;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	CUDAMallocZero(&bias_, outC_*sizeof(__half));
	if(train_){
		HeInit(weights_, weightCount_, fanIn);
		checkCUDNN(cudnnCreateTensorDescriptor(&outGradDesc_));
		checkCUDNN(cudnnCreateTensorDescriptor(&inGradDesc_));
		checkCUDNN(cudnnSetTensor4dDescriptor(outGradDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, inHeight, inWidth));
		checkCUDNN(cudnnSetTensor4dDescriptor(inGradDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, outHeight, outWidth));
		CUDAMallocZero(&gradWeights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&gradBias_, outC_*sizeof(__half));
		CUDAMallocZero(&gradOut_, gradOutSize_);
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&v_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&m_Bias_, outC_*sizeof(__half));
			CUDAMallocZero(&v_Bias_, outC_*sizeof(__half));
		}
	}
	algos_ = GetConvolutionAlgorithms(cudnnHandle_, inDesc_, filterDesc_, convDesc_, outDesc_, train);
	CUDAMallocZero(&workspace_, algos_.workspaceSize);
	*width = outWidth;
	*height = outHeight;
}
ConvLayer::~ConvLayer(){
	cudaFree(outData_);
	cudaFree(weights_);
	cudaFree(bias_);
	cudaFree(workspace_);
	checkCUDNN(cudnnDestroyTensorDescriptor(inDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc_));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasDesc_));
	if(train_){
		cudaFree(gradWeights_);
		cudaFree(gradBias_);
		cudaFree(gradOut_);
		if(useAdamW_){
			cudaFree(m_Weights_);
			cudaFree(v_Weights_);
			cudaFree(m_Bias_);
			cudaFree(v_Bias_);
		}
		checkCUDNN(cudnnDestroyTensorDescriptor(outGradDesc_));
	}
}
__half* ConvLayer::Forward(__half* data){
	inData_ = data;
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, inDesc_, data, filterDesc_, weights_, convDesc_, algos_.fwdAlgo, workspace_, algos_.workspaceSize, &beta0, outDesc_, outData_));
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, biasDesc_, bias_, &beta1, outDesc_, outData_));
	return outData_;
}
__half* ConvLayer::Backward(__half* grad){
	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &alpha, inDesc_, inData_, inGradDesc_, grad, convDesc_, algos_.bwdFilterAlgo, workspace_, algos_.workspaceSize, &beta0, filterDesc_, gradWeights_));
	checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle_, &alpha, inGradDesc_, grad, &beta0, biasDesc_, gradBias_));
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &alpha, filterDesc_, weights_, inGradDesc_, grad, convDesc_, algos_.bwdDataAlgo, workspace_, algos_.workspaceSize, &beta0, outGradDesc_, gradOut_));
	return gradOut_;
}
void ConvLayer::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, weightCount_);
		AdamWHalf(bias_, gradBias_, m_Bias_, v_Bias_, learningRate, t_, weightDecay_, outC_);
		++t_;
	} else{
		SGDHalf(weights_, gradWeights_, weightCount_, learningRate, weightDecay_);
		SGDHalf(bias_, gradBias_, outC_, learningRate, weightDecay_);
	}
}
void ConvLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
}
void ConvLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
}
void ConvLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, m_Bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(buffer, v_Bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
}
void ConvLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(m_Bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(v_Bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
}
size_t ConvLayer::GetParameterSize(){
	return weightCount_*sizeof(__half);
}
size_t ConvLayer::GetOptimizerStateSize(){
	return weightCount_*sizeof(__half);
}