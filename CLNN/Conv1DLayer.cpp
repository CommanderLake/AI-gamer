#include "Conv1DLayer.h"
#include "CuCommon.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

Conv1DLayer::Conv1DLayer(cudnnHandle_t cudnnHandle, const int batchSize, const int inputChannels, const int outputChannels, const int kernelSize, const int stride, const int dilation, int* width, std::string layerName, const bool train, const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod, const bool backpropInput) :
	cudnnHandle_(cudnnHandle), batchSize_(batchSize), inC_(inputChannels), outC_(outputChannels), inWidth_(*width), kernelSize_(kernelSize), stride_(stride), dilation_(dilation), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength), backpropInput_(backpropInput){
	if(batchSize_ <= 0 || inC_ <= 0 || outC_ <= 0 || inWidth_ <= 0 || kernelSize_ <= 0 || stride_ <= 0 || dilation_ <= 0){ throw std::invalid_argument("Invalid Conv1D layer configuration"); }
	layerName_ = layerName;
	train_ = train;
	alphaWeights_ = 1.0f/(batchSize_*gradAccumLength_);
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc_));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, 1, inWidth_));
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, outC_, inC_, 1, kernelSize_));
	const int padWidth = dilation_*(kernelSize_ - 1)/2;
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc_, 0, padWidth, 1, stride_, 1, dilation_, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolutionMathType(convDesc_, CUDNN_TENSOR_OP_MATH));
	int n, c, h;
	checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc_, inDesc_, filterDesc_, &n, &c, &h, &outWidth_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, 1, outWidth_));
	outNCHW_ = static_cast<size_t>(batchSize_)*outC_*outWidth_;
	inNCHW_ = batchSize_*inC_*inWidth_;
	weightCount_ = static_cast<size_t>(outC_)*inC_*kernelSize_;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	if(train_){
		WeightInit(weights_, static_cast<int>(weightCount_), inC_*kernelSize_, outC_*kernelSize_, weightInitMethod);
		CUDAMallocZero(&gradWeights_, weightCount_*sizeof(__half));
		if(backpropInput_){ CUDAMallocZero(&outGrad_, static_cast<size_t>(inNCHW_)*sizeof(__half)); }
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&v_Weights_, weightCount_*sizeof(__half));
		}
	}
	algos_ = GetConvolutionAlgorithms(cudnnHandle_, inDesc_, filterDesc_, convDesc_, outDesc_, train, backpropInput_);
	if(algos_.workspaceSize > 0){
		CUDAMallocZero(&workspace_, algos_.workspaceSize);
		ownsWorkspace_ = true;
	}
	*width = outWidth_;
	std::cout << layerName_ << " out width: " << outWidth_ << " kernel: " << kernelSize_ << " stride: " << stride_ << " dilation: " << dilation_ << "\n";
}

Conv1DLayer::~Conv1DLayer(){
	cudaFree(outData_);
	cudaFree(weights_);
	if(ownsWorkspace_){ cudaFree(workspace_); }
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

__half* Conv1DLayer::Forward(__half* data){
	inData_ = data;
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &one_, inDesc_, data, filterDesc_, weights_, convDesc_, algos_.fwdAlgo, workspace_, algos_.workspaceSize, &zero_, outDesc_, outData_));
	return outData_;
}

__half* Conv1DLayer::Backward(__half* grad){
	const float* betaWeights = accumCount_++%gradAccumLength_ == 0 ? &zero_ : &one_;
	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &alphaWeights_, inDesc_, inData_, outDesc_, grad, convDesc_, algos_.bwdFilterAlgo, workspace_, algos_.workspaceSize, betaWeights, filterDesc_, gradWeights_));
	if(!backpropInput_){ return nullptr; }
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &one_, filterDesc_, weights_, outDesc_, grad, convDesc_, algos_.bwdDataAlgo, workspace_, algos_.workspaceSize, &zero_, inDesc_, outGrad_));
	return outGrad_;
}

void Conv1DLayer::UpdateParameters(const float learningRate){
	if(accumCount_%gradAccumLength_ > 0) return;
	if(useAdamW_){
		++t_;
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, static_cast<int>(weightCount_));
	} else{ SGDHalf(weights_, gradWeights_, static_cast<int>(weightCount_), learningRate, weightDecay_); }
}

void Conv1DLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}

void Conv1DLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}

void Conv1DLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	file.write(reinterpret_cast<char*>(&t_), sizeof(int));
}

void Conv1DLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}

size_t Conv1DLayer::GetParameterSize(){ return weightCount_*sizeof(__half); }
size_t Conv1DLayer::GetOptimizerStateSize(){ return weightCount_*sizeof(__half); }
void Conv1DLayer::SetTrain(const bool enable){ train_ = enable; }

void Conv1DLayer::ReleaseWorkspace(){
	if(ownsWorkspace_){ cudaFree(workspace_); }
	workspace_ = nullptr;
	ownsWorkspace_ = false;
}

void Conv1DLayer::UseSharedWorkspace(void* workspace, const size_t workspaceSize){
	if(workspaceSize < algos_.workspaceSize){ throw std::invalid_argument("Shared Conv1D workspace is too small"); }
	ReleaseWorkspace();
	workspace_ = workspace;
}

void Conv1DLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!useAdamW_ || !train_) return;
	halfTasks.push_back({weights_, gradWeights_, m_Weights_, v_Weights_, static_cast<int>(weightCount_), weightDecay_});
}

Conv1DLayer::ConvolutionAlgorithms Conv1DLayer::GetConvolutionAlgorithms(const cudnnHandle_t cudnnHandle, const cudnnTensorDescriptor_t xDesc, const cudnnFilterDescriptor_t wDesc, const cudnnConvolutionDescriptor_t convDesc, const cudnnTensorDescriptor_t yDesc, const bool isTraining, const bool backpropInput){
	ConvolutionAlgorithms algorithms{};
	cudnnConvolutionFwdAlgoPerf_t fwdAlgoPerf[10];
	int returnedAlgoCount = 0;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle, xDesc, wDesc, convDesc, yDesc, 10, &returnedAlgoCount, fwdAlgoPerf));
	algorithms.fwdAlgo = fwdAlgoPerf[0].algo;
	algorithms.workspaceSize = fwdAlgoPerf[0].memory;
	if(isTraining){
		if(backpropInput){
			cudnnConvolutionBwdDataAlgoPerf_t bwdDataAlgoPerf[10];
			checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle, wDesc, yDesc, convDesc, xDesc, 10, &returnedAlgoCount, bwdDataAlgoPerf));
			algorithms.bwdDataAlgo = bwdDataAlgoPerf[0].algo;
			algorithms.workspaceSize = std::max(algorithms.workspaceSize, bwdDataAlgoPerf[0].memory);
		}
		cudnnConvolutionBwdFilterAlgoPerf_t bwdFilterAlgoPerf[10];
		checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle, xDesc, yDesc, convDesc, wDesc, 10, &returnedAlgoCount, bwdFilterAlgoPerf));
		algorithms.bwdFilterAlgo = bwdFilterAlgoPerf[0].algo;
		algorithms.workspaceSize = std::max(algorithms.workspaceSize, bwdFilterAlgoPerf[0].memory);
	} else{
		algorithms.bwdDataAlgo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
		algorithms.bwdFilterAlgo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
	}
	return algorithms;
}
