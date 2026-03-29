#include "ActionHead.h"
#include "common.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "GELULayer.h"
#include "FCLayer.h"
#undef min
#undef max
ActionHead::ActionHead(const int batchSize, const int patchRows, const int patchCols, const int embedSize, const std::string layerName, const bool train, const float weightDecay, const int gradAccumLength) : batchSize_(batchSize), nTokens_(patchRows*patchCols), embedSize_(embedSize), patchRows_(patchRows), patchCols_(patchCols), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	inC_ = embedSize_*patchRows_*patchCols_;
	constexpr auto hiddenC = 1024;
	layers_.push_back(new FCLayer(batchSize_, inC_, hiddenC, "Action FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	layers_.push_back(new GELULayer(batchSize_, hiddenC, 1, 1, "Action GELU"));
	layers_.push_back(new FCLayer(batchSize_, hiddenC, NUM_CTRLS_, "Action FC 2", train_, weightDecay_, gradAccumLength_, Xavier, true));
}
ActionHead::~ActionHead(){
	for(const auto* layer : layers_) delete layer;
	layers_.clear();
}
__half* ActionHead::Forward(__half* data){
	for(auto* layer : layers_){
		data = layer->Forward(data);
	}
	return data;
}
__half* ActionHead::Backward(__half* grad){
	for(int i = static_cast<int>(layers_.size()); --i >= 0;){
		grad = layers_[i]->Backward(grad);
	}
	return grad;
}
void ActionHead::UpdateParameters(const float learningRate){
	for(auto* layer : layers_){ layer->UpdateParameters(learningRate); }
}
void ActionHead::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : layers_){ layer->SaveParameters(file, buffer); }
}
void ActionHead::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : layers_){ layer->LoadParameters(file, buffer); }
}
void ActionHead::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : layers_){ layer->SaveOptimizerState(file, buffer); }
}
void ActionHead::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : layers_){ layer->LoadOptimizerState(file, buffer); }
}
size_t ActionHead::GetParameterSize(){
	size_t maxSize = 0;
	for(auto* layer : layers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	return maxSize;
}
size_t ActionHead::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(auto* layer : layers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	return maxSize;
}
void ActionHead::SetTrain(const bool enable){
	train_ = enable;
	for(auto* layer : layers_){ layer->SetTrain(enable); }
}
void ActionHead::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	for(auto* layer : layers_){ layer->CollectAdamWTasks(halfTasks, floatTasks); }
}