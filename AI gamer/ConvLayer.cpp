#include "ConvLayer.h"
#include <iostream>
ConvLayer::ConvLayer(cudnnHandle_t cudnnHandle, int batchSize, int inputChannels, int outputChannels, int filterSize, int stride, int* height, int* width, const char* layerName, bool train, float weightDecay, int gradAccumLength) : cudnnHandle_(cudnnHandle),
	batchSize_(batchSize), outC_(outputChannels), inC_(inputChannels), inHeight_(*height), inWidth_(*width), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	alphaWeights_ = 1.0f/gradAccumLength_;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc_));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, inHeight_, inWidth_));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, outC_, inC_, filterSize, filterSize));
	auto [padH, padW] = Padding(inHeight_, inWidth_, filterSize, stride);
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc_, padH, padW, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolutionMathType(convDesc_, CUDNN_TENSOR_OP_MATH)); //S
	int n, c;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc_, inDesc_, filterDesc_, &n, &c, &outHeight_, &outWidth_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, outHeight_, outWidth_));
	outNCHW_ = outWidth_*outHeight_*outC_*batchSize_;
	inNCHW_ = inWidth_*inHeight_*inC_*batchSize_;
	const auto fanIn = inC_*filterSize*filterSize;
	weightCount_ = outC_*fanIn;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	if(train_){
		WeightInit(weights_, weightCount_, fanIn, outC_, He);
		CUDAMallocZero(&gradWeights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&outGrad_, inNCHW_*sizeof(__half));
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&v_Weights_, weightCount_*sizeof(__half));
		}
	}
	algos_ = GetConvolutionAlgorithms(cudnnHandle_, inDesc_, filterDesc_, convDesc_, outDesc_, train);
	CUDAMallocZero(&workspace_, algos_.workspaceSize);
	*width = outWidth_;
	*height = outHeight_;
}
ConvLayer::~ConvLayer(){
	cudaFree(outData_);
	cudaFree(weights_);
	cudaFree(workspace_);
	checkCUDNN(cudnnDestroyTensorDescriptor(inDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc_));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc_));
	if(train_){
		cudaFree(gradWeights_);
		cudaFree(outGrad_);
		if(useAdamW_){
			cudaFree(m_Weights_);
			cudaFree(v_Weights_);
		}
	}
}
__half* ConvLayer::Forward(__half* data){
	inData_ = data;
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha_, inDesc_, data, filterDesc_, weights_, convDesc_, algos_.fwdAlgo, workspace_, algos_.workspaceSize, &beta0_, outDesc_, outData_));
	return outData_;
}
__half* ConvLayer::Backward(__half* grad){
	const float* betaWeights = accumCount_++%gradAccumLength_==0 ? &beta0_ : &beta1_;
	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &alphaWeights_, inDesc_, inData_, outDesc_, grad, convDesc_, algos_.bwdFilterAlgo, workspace_, algos_.workspaceSize, betaWeights, filterDesc_, gradWeights_));
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &alpha_, filterDesc_, weights_, outDesc_, grad, convDesc_, algos_.bwdDataAlgo, workspace_, algos_.workspaceSize, &beta0_, inDesc_, outGrad_));
	return outGrad_;
}
void ConvLayer::UpdateParameters(const float learningRate){
	if(accumCount_%gradAccumLength_>0) return;
	if(useAdamW_){
		++t_;
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, weightCount_);
	} else{
		SGDHalf(weights_, gradWeights_, weightCount_, learningRate, weightDecay_);
	}
}
void ConvLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}
void ConvLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
void ConvLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}
void ConvLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
size_t ConvLayer::GetParameterSize(){
	return weightCount_*sizeof(__half);
}
size_t ConvLayer::GetOptimizerStateSize(){
	return weightCount_*sizeof(__half);
}
void ConvLayer::SetTrain(bool enable){
	int bs;
	if(enable){
		train_ = true;
		bs = batchSize_;
	} else{
		train_ = false;
		bs = 1;
	}
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, inC_, inHeight_, inWidth_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, outC_, outHeight_, outWidth_));
}
std::pair<int, int> ConvLayer::Padding(const int imageHeight, const int imageWidth, const int kernelSize, const int stride){
	auto computePadding = [](const int imageSize, const int kernelSize, int stride){
		const int coverage = (imageSize - kernelSize)/stride*stride + (kernelSize - 1);
		return std::max(0, coverage - (imageSize - 1));
	};
	return {computePadding(imageHeight, kernelSize, stride), computePadding(imageWidth, kernelSize, stride)};
}