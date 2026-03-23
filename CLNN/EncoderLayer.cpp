#include "NNCommon.h"
#include "CuCommon.cuh"
#include "EncoderLayer.h"
#include "LayerNorm.h"
#include "WmmaAttentionLayer.h"
#include "FCLayer.h"
#include "GELULayer.h"
#include "Dropout.h"
#include <algorithm>
EncoderLayer::EncoderLayer(const cudnnHandle_t cudnnHandle, const int batchSize, const int tokens, const int embedDim, const int ffDim, const int numHeads, std::string layerName, const bool train, const float weightDecay, const int gradAccumLength) : cudnnHandle_(cudnnHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), ffDim_(ffDim), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*tokens_*embedDim_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_*tokens_, embedDim_, 1, 1));
	layers_.push_back(new LayerNorm(batchSize_*tokens_, embedDim_, 1, 1, "Norm1", train));
	layers_.push_back(new WmmaAttentionLayer(cudnnHandle_, batchSize_, tokens_, embedDim_, numHeads, "Attention", train_, weightDecay, gradAccumLength_, Xavier));
	layers_.push_back(new Dropout(cudnnHandle_, 0.1f, batchSize_*tokens_, embedDim_, 1, 1, "Attn_Dropout", train));
	layers_.push_back(new LayerNorm(batchSize_*tokens_, embedDim_, 1, 1, "Norm2", train));
	layers_.push_back(new FCLayer(batchSize_*tokens_, embedDim_, ffDim_, "FC1", train_, weightDecay, gradAccumLength_, Xavier, true));
	layers_.push_back(new GELULayer(batchSize_*tokens_, ffDim_, 1, 1, "GELU"));
	layers_.push_back(new FCLayer(batchSize_*tokens_, ffDim_, embedDim_, "FC2", train_, weightDecay, gradAccumLength_, Xavier, true));
	layers_.push_back(new Dropout(cudnnHandle_, 0.1f, batchSize_*tokens_, embedDim_, 1, 1, "FF_Dropout", train));
}
EncoderLayer::~EncoderLayer(){
	for(const auto layer : layers_) delete layer;
	layers_.clear();
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
}
__half* EncoderLayer::Forward(__half* data){
	const auto* residual1 = data;
	//std::cout << "\n" << layers_[0]->layerName_ << " ";
	data = layers_[0]->Forward(data);
	//SummarizeHalfDevice(data, layers_[0]->outNCHW_, "data");
	//std::cout << "\n" << layers_[1]->layerName_ << " ";
	data = layers_[1]->Forward(data);
	//SummarizeHalfDevice(data, layers_[1]->outNCHW_, "data");
	//std::cout << "\n" << layers_[2]->layerName_ << " ";
	data = layers_[2]->Forward(data);
	//SummarizeHalfDevice(data, layers_[2]->outNCHW_, "data");
	//std::cout << "\n" << layers_[3]->layerName_ << " ";
	AddTensor(mixFwd_, data, mixFwd_, residual1, static_cast<int>(outNCHW_));
	const auto* residual2 = data;
	data = layers_[3]->Forward(data);
	//SummarizeHalfDevice(data, layers_[3]->outNCHW_, "data");
	//std::cout << "\n" << layers_[4]->layerName_ << " ";
	data = layers_[4]->Forward(data);
	//SummarizeHalfDevice(data, layers_[4]->outNCHW_, "data");
	//std::cout << "\n" << layers_[5]->layerName_ << " ";
	data = layers_[5]->Forward(data);
	//SummarizeHalfDevice(data, layers_[5]->outNCHW_, "data");
	//std::cout << "\n" << layers_[6]->layerName_ << " ";
	data = layers_[6]->Forward(data);
	//SummarizeHalfDevice(data, layers_[6]->outNCHW_, "data");
	//std::cout << "\n" << layers_[7]->layerName_ << " ";
	data = layers_[7]->Forward(data);
	//SummarizeHalfDevice(data, layers_[7]->outNCHW_, "data");
	AddTensor(mixFwd_, data, mixFwd_, residual2, static_cast<int>(outNCHW_));
	return data;
}
__half* EncoderLayer::Backward(__half* grad){
	const auto* residual2 = grad;
	grad = layers_[7]->Backward(grad);
	grad = layers_[6]->Backward(grad);
	grad = layers_[5]->Backward(grad);
	grad = layers_[4]->Backward(grad);
	grad = layers_[3]->Backward(grad);
	AddTensor(mixBwd_, grad, mixBwd_, residual2, static_cast<int>(outNCHW_));
	const auto* residual1 = grad;
	grad = layers_[2]->Backward(grad);
	grad = layers_[1]->Backward(grad);
	grad = layers_[0]->Backward(grad);
	AddTensor(mixBwd_, grad, mixBwd_, residual1, static_cast<int>(outNCHW_));
	return grad;
}
void EncoderLayer::UpdateParameters(const float lr){ for(const auto layer : layers_){ layer->UpdateParameters(lr); } }
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
	for(const auto layer : layers_){ layer->SetTrain(enable); }
}

void EncoderLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	for(const auto layer : layers_){ layer->CollectAdamWTasks(halfTasks, floatTasks); }
}
