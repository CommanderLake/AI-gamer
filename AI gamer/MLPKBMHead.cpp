#include "MLPKBMHead.h"
#include "common.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "GELULayer.h"
#include "Dropout.h"
#include "FCLayer.h"
#include "ViewerLayer.h"
#undef min
#undef max
MLPKBMHead::MLPKBMHead(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int embedSize, std::string layerName, const bool train, const float weightDecay, const int gradAccumLength) : cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(batchSize), embedSize_(embedSize), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	CUDAMallocZero(&predictions_, batchSize_*NUM_CTRLS_*sizeof(__half));
	constexpr auto hiddenC = 2048;
	buttonLayers_.push_back(new FCLayer(cublas_, batchSize_, embedSize_, hiddenC, "Buttons FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	buttonLayers_.push_back(new GELULayer(batchSize_, hiddenC, 1, 1, "Buttons GELU"));
	buttonLayers_.push_back(new Dropout(cudnn_, 0.2f, batchSize_, hiddenC, 1, 1, "Buttons Drop", train_));
	buttonLayers_.push_back(new FCLayer(cublas_, batchSize_, hiddenC, NUM_BUTS_, "Buttons FC 2", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new FCLayer(cublas_, batchSize_, embedSize_, hiddenC, "Axes FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new GELULayer(batchSize_, hiddenC, 1, 1, "Axes GELU"));
	axisLayers_.push_back(new Dropout(cudnn_, 0.2f, batchSize_, hiddenC, 1, 1, "Axes Drop", train_));
	axisLayers_.push_back(new FCLayer(cublas_, batchSize_, hiddenC, NUM_AXES_, "Axes FC 2", train_, weightDecay_, gradAccumLength_, Xavier, false));
}
MLPKBMHead::~MLPKBMHead(){
	cudaFree(predictions_);
	for(const auto* layer : axisLayers_) delete layer;
	for(const auto* layer : buttonLayers_) delete layer;
	axisLayers_.clear();
	buttonLayers_.clear();
}
__half* MLPKBMHead::Forward(__half* data){
	auto buttonData = data;
	for(auto* layer : buttonLayers_){ buttonData = layer->Forward(buttonData); }
	auto axisData = data;
	for(auto* layer : axisLayers_){ axisData = layer->Forward(axisData); }
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*batchSize_);
	return predictions_;
}
__half* MLPKBMHead::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad + NUM_BUTS_*batchSize_;
	for(int i = static_cast<int>(buttonLayers_.size()); --i >= 0;){ buttonGrad = buttonLayers_[i]->Backward(buttonGrad); }
	for(int i = static_cast<int>(axisLayers_.size()); --i >= 0;){ axisGrad = axisLayers_[i]->Backward(axisGrad); }
	AddTensor(1.0f, buttonGrad, 1.0f, axisGrad, batchSize_*embedSize_);
	return buttonGrad;
}
void MLPKBMHead::UpdateParameters(const float learningRate){
	for(auto* layer : buttonLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : axisLayers_){ layer->UpdateParameters(learningRate); }
}
void MLPKBMHead::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : buttonLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveParameters(file, buffer); }
}
void MLPKBMHead::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : buttonLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadParameters(file, buffer); }
}
void MLPKBMHead::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : buttonLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveOptimizerState(file, buffer); }
}
void MLPKBMHead::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : buttonLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadOptimizerState(file, buffer); }
}
size_t MLPKBMHead::GetParameterSize(){
	size_t maxSize = 0;
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	return maxSize;
}
size_t MLPKBMHead::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	return maxSize;
}
void MLPKBMHead::SetTrain(const bool enable){
	train_ = enable;
	for(auto* layer : buttonLayers_){ layer->SetTrain(enable); }
	for(auto* layer : axisLayers_){ layer->SetTrain(enable); }
}
