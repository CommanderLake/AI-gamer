#include "SwinBlockLayer.h"
#include "CuCommon.cuh"
#include "Dropout.h"
#include "FCLayer.h"
#include "GELULayer.h"
#include "LayerNorm.h"
#include "WindowAttentionLayer.h"
#include <algorithm>

SwinBlockLayer::SwinBlockLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int tokens, const int embedDim, const int ffDim, const int numHeads, const int patchRows, const int patchCols, const int windowSize, const int shiftSize, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), ffDim_(ffDim), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = static_cast<size_t>(batchSize_) * tokens_ * embedDim_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_ * tokens_, embedDim_, 1, 1));
	layers_.push_back(new LayerNorm(batchSize_ * tokens_, embedDim_, 1, 1, "SwinNorm1", train));
	layers_.push_back(new WindowAttentionLayer(cudnnHandle_, cublasHandle_, batchSize_, tokens_, embedDim_, numHeads, patchRows, patchCols, windowSize, shiftSize, "WindowAttention", train_, weightDecay, gradAccumLength_, Xavier));
	layers_.push_back(new Dropout(cudnnHandle_, 0.1f, batchSize_ * tokens_, embedDim_, 1, 1, "SwinAttnDropout", train));
	layers_.push_back(new LayerNorm(batchSize_ * tokens_, embedDim_, 1, 1, "SwinNorm2", train));
	layers_.push_back(new FCLayer(cublasHandle_, batchSize_ * tokens_, embedDim_, ffDim_, "SwinFC1", train_, weightDecay, gradAccumLength_, Xavier));
	layers_.push_back(new GELULayer(batchSize_ * tokens_, ffDim_, 1, 1, "SwinGELU"));
	layers_.push_back(new FCLayer(cublasHandle_, batchSize_ * tokens_, ffDim_, embedDim_, "SwinFC2", train_, weightDecay, gradAccumLength_, Xavier));
	layers_.push_back(new Dropout(cudnnHandle_, 0.1f, batchSize_ * tokens_, embedDim_, 1, 1, "SwinFFDropout", train));
}

SwinBlockLayer::~SwinBlockLayer(){
	for(const auto layer : layers_){
		delete layer;
	}
	layers_.clear();
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
}

__half* SwinBlockLayer::Forward(__half* data){
	const auto* residual1 = data;
	data = layers_[0]->Forward(data);
	data = layers_[1]->Forward(data);
	data = layers_[2]->Forward(data);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixFwd_, outDesc_, residual1, &mixFwd_, outDesc_, data));
	const auto* residual2 = data;
	data = layers_[3]->Forward(data);
	data = layers_[4]->Forward(data);
	data = layers_[5]->Forward(data);
	data = layers_[6]->Forward(data);
	data = layers_[7]->Forward(data);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixFwd_, outDesc_, residual2, &mixFwd_, outDesc_, data));
	return data;
}

__half* SwinBlockLayer::Backward(__half* grad){
	const auto* residual2 = grad;
	grad = layers_[7]->Backward(grad);
	grad = layers_[6]->Backward(grad);
	grad = layers_[5]->Backward(grad);
	grad = layers_[4]->Backward(grad);
	grad = layers_[3]->Backward(grad);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixBwd_, outDesc_, residual2, &mixBwd_, outDesc_, grad));
	const auto* residual1 = grad;
	grad = layers_[2]->Backward(grad);
	grad = layers_[1]->Backward(grad);
	grad = layers_[0]->Backward(grad);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixBwd_, outDesc_, residual1, &mixBwd_, outDesc_, grad));
	return grad;
}

void SwinBlockLayer::UpdateParameters(const float learningRate){
	for(const auto layer : layers_){
		layer->UpdateParameters(learningRate);
	}
}

void SwinBlockLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(const auto layer : layers_){
		layer->SaveParameters(file, buffer);
	}
}

void SwinBlockLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(const auto layer : layers_){
		layer->LoadParameters(file, buffer);
	}
}

void SwinBlockLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(const auto layer : layers_){
		layer->SaveOptimizerState(file, buffer);
	}
}

void SwinBlockLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(const auto layer : layers_){
		layer->LoadOptimizerState(file, buffer);
	}
}

size_t SwinBlockLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(const auto layer : layers_){
		maxSize = std::max(maxSize, layer->GetParameterSize());
	}
	return maxSize;
}

size_t SwinBlockLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(const auto layer : layers_){
		maxSize = std::max(maxSize, layer->GetOptimizerStateSize());
	}
	return maxSize;
}

void SwinBlockLayer::SetTrain(const bool enable){
	train_ = enable;
	for(const auto layer : layers_){
		layer->SetTrain(enable);
	}
}
