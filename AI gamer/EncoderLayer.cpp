#include "common.h"
#include "CuCommon.cuh"
#include "EncoderLayer.h"
#include "BatchNorm.h"
#include "LayerNorm.h"
#include "WmmaAttentionLayer.h"
#include "FCLayer.h"
#include "GELULayer.h"
#include "Dropout.h"
#include <algorithm>
EncoderLayer::EncoderLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int tokens, const int embedDim, const int ffDim, const int numHeads, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength, const int maxTokens) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), ffDim_(ffDim), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*tokens_*embedDim_;
	checkCUDNN(cudnnCreateTensorDescriptor(&tensorDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_*tokens_, embedDim_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_*tokens_, embedDim_, 1, 1));
	layers_.push_back(new LayerNorm(batchSize_*tokens_, embedDim_, 1, 1, "Norm1", train, weightDecay));
	layers_.push_back(new WmmaAttentionLayer(cudnnHandle_, cublasHandle_, batchSize_, tokens_, embedDim_, numHeads, "Attention", train_, weightDecay, gradAccumLength_, Xavier, maxTokens));
	layers_.push_back(new Dropout(cudnnHandle_, 0.1f, batchSize_*tokens_, embedDim_, 1, 1, "Attn_Dropout", train));
	layers_.push_back(new LayerNorm(batchSize_*tokens_, embedDim_, 1, 1, "Norm2", train, weightDecay));
	layers_.push_back(new FCLayer(cudnnHandle_, cublasHandle_, batchSize_*tokens_, embedDim_, ffDim_, "FC1", train_, weightDecay, gradAccumLength_, Xavier));
	layers_.push_back(new GELULayer(batchSize_*tokens_, ffDim_, 1, 1, "GELU"));
	layers_.push_back(new Dropout(cudnnHandle_, 0.1f, batchSize_*tokens_, ffDim_, 1, 1, "FF_Dropout", train));
	layers_.push_back(new FCLayer(cudnnHandle_, cublasHandle_, batchSize_*tokens_, ffDim_, embedDim_, "FC2", train_, weightDecay, gradAccumLength_, Xavier));
}
EncoderLayer::~EncoderLayer(){
	for(const auto layer : layers_){ delete layer; }
	layers_.clear();
	checkCUDNN(cudnnDestroyTensorDescriptor(tensorDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
}
__half* EncoderLayer::Forward(__half* data){
	const __half* residual1 = data;
	for(size_t i = 0; i < 3; ++i){
		//std::cout << "\n" << layers_[i]->layerName_ << " ";
		data = layers_[i]->Forward(data);
		//SummarizeHalfDevice(data, layers_[i]->outNCHW_, "data");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &blendFwd_, tensorDesc_, residual1, &blendFwd_, tensorDesc_, data));
	//std::cout << "\n" << layers_[3]->layerName_ << " ";
	data = layers_[3]->Forward(data);
	//SummarizeHalfDevice(data, layers_[3]->outNCHW_, "data");
	const __half* residual2 = data;
	for(size_t i = 4; i < layers_.size(); ++i){
		//std::cout << "\n" << layers_[i]->layerName_ << " ";
		data = layers_[i]->Forward(data);
		//SummarizeHalfDevice(data, layers_[i]->outNCHW_, "data");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &blendFwd_, tensorDesc_, residual2, &blendFwd_, tensorDesc_, data));
	return data;
}
__half* EncoderLayer::Backward(__half* grad){
	const __half* gradAdd2 = grad;
	for(int i = layers_.size(); --i > 3;){
		//std::cout << "\n" << layers_[i]->layerName_ << " ";
		grad = layers_[i]->Backward(grad);
		//SummarizeHalfDevice(grad, layers_[i]->outNCHW_, "gradient");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &blendBwd_, tensorDesc_, gradAdd2, &blendBwd_, tensorDesc_, grad));
	//std::cout << "\n" << layers_[3]->layerName_ << " ";
	grad = layers_[3]->Backward(grad);
	//SummarizeHalfDevice(grad, layers_[3]->outNCHW_, "gradient");
	const __half* gradAdd1 = grad;
	for(int i = 3; --i >= 0;){
		//std::cout << "\n" << layers_[i]->layerName_ << " ";
		grad = layers_[i]->Backward(grad);
		//SummarizeHalfDevice(grad, layers_[i]->outNCHW_, "gradient");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &blendBwd_, tensorDesc_, gradAdd1, &blendBwd_, tensorDesc_, grad));
	return grad;
}
void EncoderLayer::UpdateParameters(float learningRate){ for(const auto layer : layers_){ layer->UpdateParameters(learningRate); } }
void EncoderLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){ for(const auto layer : layers_){ layer->SaveParameters(file, buffer); } }
void EncoderLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){ for(const auto layer : layers_){ layer->LoadParameters(file, buffer); } }
void EncoderLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){ for(const auto layer : layers_){ layer->SaveOptimizerState(file, buffer); } }
void EncoderLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){ for(const auto layer : layers_){ layer->LoadOptimizerState(file, buffer); } }
size_t EncoderLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(const auto layer : layers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	return maxSize;
}
size_t EncoderLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(const auto layer : layers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	return maxSize;
}
void EncoderLayer::SetTrain(bool enable){
	train_ = enable;
	const int bs = enable ? batchSize_ : 1;
	checkCUDNN(cudnnSetTensor4dDescriptor(tensorDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs*tokens_, embedDim_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs*tokens_, embedDim_, 1, 1));
	for(const auto layer : layers_){ layer->SetTrain(enable); }
}
void EncoderLayer::SetDropout(bool enable){ for(const auto layer : layers_){ layer->SetDropout(enable); } }