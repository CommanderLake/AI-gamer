#include "CustomOutLayer.h"
#include "common.h"
#include "CuCommon.cuh"
#include "FCLayer.h"
#include "GELULayer.h"
#include "SigmoidLayer.h"
//#include "ViewerLayer.h"
CustomOutLayer::CustomOutLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int seqLength, const int inputSize, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength) :
	cudnn_(cudnnHandle), cublas_(cublasHandle), ogbs_(batchSize), batchSize_(batchSize), seqLength_(seqLength), inC_(inputSize), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_*seqLength_, inputSize, 1, 1));
	constexpr int hiddenDim = 256;
	buttonLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_*seqLength_, inputSize, NUM_BUTS_, "Buts_FC1", train, weightDecay, gradAccumLength_, Xavier, 1.0f, true));
	buttonLayers_.push_back(new SigmoidLayer(batchSize_*seqLength_, NUM_BUTS_, NUM_BUTS_, "Buts_Sigmoid"));
	axisLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_*seqLength_, inputSize, hiddenDim, "Axes_FC1", train, weightDecay, gradAccumLength_, Xavier, 1.0f, true));
	axisLayers_.push_back(new GELULayer(batchSize_*seqLength_, hiddenDim, 1, 1, "GELU"));
	axisLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_*seqLength_, hiddenDim, NUM_AXES_, "Axes_FC_Out", train, weightDecay, gradAccumLength_, Xavier, 0.125f, true));
	CUDAMallocZero(&predictions_, batchSize_*seqLength_*NUM_CTRLS_*sizeof(__half));
}
CustomOutLayer::~CustomOutLayer(){
	cudaFree(predictions_);
	for(const auto layer : axisLayers_){
		delete layer;
	}
	for(const auto layer : buttonLayers_){
		delete layer;
	}
	axisLayers_.clear();
	buttonLayers_.clear();
	cudnnDestroyTensorDescriptor(inDesc_);
}
__half* CustomOutLayer::Forward(__half* data){
	auto buttonData = data;
	auto axisData = data;
	for(int i = 0; i<buttonLayers_.size(); ++i){
		//std::cout << "\n" << buttonLayers_[i]->layerName_ << " ";
		buttonData = buttonLayers_[i]->Forward(buttonData);
		//SummarizeHalfDevice(buttonData, buttonLayers_[i]->outNCHW_, "buttonData");
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		//std::cout << "\n" << axisLayers_[i]->layerName_ << " ";
		axisData = axisLayers_[i]->Forward(axisData);
		//SummarizeHalfDevice(axisData, axisLayers_[i]->outNCHW_, "axisData");
	}
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*batchSize_*seqLength_);
	return predictions_;
}
__half* CustomOutLayer::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad+NUM_BUTS_*batchSize_*seqLength_;
	for(int i = buttonLayers_.size(); --i>=0;){
		//std::cout << "\n" << buttonLayers_[i]->layerName_ << " ";
		buttonGrad = buttonLayers_[i]->Backward(buttonGrad);
		//SummarizeHalfDevice(buttonGrad, buttonLayers_[i]->outNCHW_, "buttonGrad");
	}
	for(int i = axisLayers_.size(); --i>=0;){
		//std::cout << "\n" << axisLayers_[i]->layerName_ << " ";
		axisGrad = axisLayers_[i]->Backward(axisGrad);
		//SummarizeHalfDevice(axisGrad, axisLayers_[i]->outNCHW_, "axisGrad");
	}
	checkCUDNN(cudnnAddTensor(cudnn_, &alpha, inDesc_, axisGrad, &alpha, inDesc_, buttonGrad));
	return buttonGrad;
}
void CustomOutLayer::UpdateParameters(const float learningRate){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->UpdateParameters(learningRate); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->UpdateParameters(learningRate); }
}
void CustomOutLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->SaveParameters(file, buffer); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->SaveParameters(file, buffer); }
}
void CustomOutLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->LoadParameters(file, buffer); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->LoadParameters(file, buffer); }
}
void CustomOutLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->SaveOptimizerState(file, buffer); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->SaveOptimizerState(file, buffer); }
}
void CustomOutLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->LoadOptimizerState(file, buffer); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->LoadOptimizerState(file, buffer); }
}
size_t CustomOutLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(int i = 0; i<buttonLayers_.size(); ++i){ maxSize = std::max(maxSize, buttonLayers_[i]->GetParameterSize()); }
	for(int i = 0; i<axisLayers_.size(); ++i){ maxSize = std::max(maxSize, axisLayers_[i]->GetParameterSize()); }
	return maxSize;
}
size_t CustomOutLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(int i = 0; i<buttonLayers_.size(); ++i){ maxSize = std::max(maxSize, buttonLayers_[i]->GetOptimizerStateSize()); }
	for(int i = 0; i<axisLayers_.size(); ++i){ maxSize = std::max(maxSize, axisLayers_[i]->GetOptimizerStateSize()); }
	return maxSize;
}
void CustomOutLayer::SetFineTune(const bool enable){
	if(enable){
		train_ = true;
		batchSize_ = ogbs_;
	} else{
		train_ = false;
		batchSize_ = 1;
	}
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, 1, 1));
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->SetFineTune(enable); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->SetFineTune(enable); }
}
void CustomOutLayer::SetDropout(const bool enable){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->SetDropout(enable); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->SetDropout(enable); }
}