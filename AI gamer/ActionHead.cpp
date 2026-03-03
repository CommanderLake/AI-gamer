#include "ActionHead.h"
#include "common.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "GELULayer.h"
#include "FCLayer.h"
#undef min
#undef max
ActionHead::ActionHead(const cudnnHandle_t cudnnHandle, const int batchSize, const int patchRows, const int patchCols, const int embedSize, const std::string layerName, const bool train, const float weightDecay, const int gradAccumLength) : cudnn_(cudnnHandle), batchSize_(batchSize), nTokens_(patchRows*patchCols), embedSize_(embedSize), patchRows_(patchRows), patchCols_(patchCols), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	CUDAMallocZero(&predictions_, batchSize_*NUM_CTRLS_*sizeof(__half));
	inC_ = embedSize_*patchRows_*patchCols_;
	constexpr auto hiddenC = 1024;
	buttonLayers_.push_back(new FCLayer(batchSize_, inC_, hiddenC, "Buttons FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	buttonLayers_.push_back(new GELULayer(batchSize_, hiddenC, 1, 1, "Buttons GELU"));
	buttonLayers_.push_back(new FCLayer(batchSize_, hiddenC, NUM_BUTS_, "Buttons FC 2", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new FCLayer(batchSize_, inC_, hiddenC, "Axes FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new GELULayer(batchSize_, hiddenC, 1, 1, "Axes GELU"));
	axisLayers_.push_back(new FCLayer(batchSize_, hiddenC, NUM_AXES_, "Axes FC 2", train_, weightDecay_, gradAccumLength_, Xavier, false));
	//axisLayers_.push_back(new AsinhLayer(batchSize_, NUM_AXES_, 1, 1, static_cast<int>(AXIS_SCALE_), "Axes Asinh"));
}
ActionHead::~ActionHead(){
	cudaFree(predictions_);
	for(const auto* layer : axisLayers_) delete layer;
	for(const auto* layer : buttonLayers_) delete layer;
	axisLayers_.clear();
	buttonLayers_.clear();
}
__half* ActionHead::Forward(__half* data){
	auto buttonData = data;
	for(auto* layer : buttonLayers_){ buttonData = layer->Forward(buttonData); }
	auto axisData = data;
	for(auto* layer : axisLayers_){ axisData = layer->Forward(axisData); }
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*batchSize_);
	return predictions_;
}
__half* ActionHead::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad + NUM_BUTS_*batchSize_;
	for(int i = static_cast<int>(buttonLayers_.size()); --i >= 0;){ buttonGrad = buttonLayers_[i]->Backward(buttonGrad); }
	for(int i = static_cast<int>(axisLayers_.size()); --i >= 0;){ axisGrad = axisLayers_[i]->Backward(axisGrad); }
	AddTensor(1.0f, buttonGrad, 1.0f, axisGrad, batchSize_*inC_);
	return buttonGrad;
}
void ActionHead::UpdateParameters(const float learningRate){
	for(auto* layer : buttonLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : axisLayers_){ layer->UpdateParameters(learningRate); }
}
void ActionHead::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : buttonLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveParameters(file, buffer); }
}
void ActionHead::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : buttonLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadParameters(file, buffer); }
}
void ActionHead::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : buttonLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveOptimizerState(file, buffer); }
}
void ActionHead::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : buttonLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadOptimizerState(file, buffer); }
}
size_t ActionHead::GetParameterSize(){
	size_t maxSize = 0;
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	return maxSize;
}
size_t ActionHead::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	return maxSize;
}
void ActionHead::SetTrain(const bool enable){
	train_ = enable;
	for(auto* layer : buttonLayers_){ layer->SetTrain(enable); }
	for(auto* layer : axisLayers_){ layer->SetTrain(enable); }
}

void ActionHead::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	for(auto* layer : buttonLayers_){ layer->CollectAdamWTasks(halfTasks, floatTasks); }
	for(auto* layer : axisLayers_){ layer->CollectAdamWTasks(halfTasks, floatTasks); }
}

