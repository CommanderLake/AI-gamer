#include "ActionHead.h"
#include "common.h"
#include "ConvLayer.h"
#include "GELULayer.h"
#include "FCLayer.h"
#include <algorithm>
#include <stdexcept>
#undef min
#undef max
ActionHead::ActionHead(const int batchSize, const int patchRows, const int patchCols, const int embedSize, const std::string layerName, const bool train, const float weightDecay, const int gradAccumLength, const int temporalLength, const TemporalOutputPolicy outputPolicy) : batchSize_(batchSize), nTokens_(patchRows*patchCols), embedSize_(embedSize), temporalLength_(std::max(1, temporalLength)), patchRows_(patchRows), patchCols_(patchCols), outputPolicy_(outputPolicy), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	if(batchSize_%temporalLength_ != 0){ throw std::invalid_argument("ActionHead batchSize must be divisible by temporalLength"); }
	baseBatchSize_ = batchSize_/temporalLength_;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	inC_ = embedSize_*patchRows_*patchCols_;
	outFeatureSize_ = NUM_CTRLS_;
	constexpr auto hiddenC = 1024;
	layers_.push_back(new FCLayer(baseBatchSize_, inC_, hiddenC, "Action FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	layers_.push_back(new GELULayer(baseBatchSize_, hiddenC, 1, 1, "Action GELU"));
	layers_.push_back(new FCLayer(baseBatchSize_, hiddenC, NUM_CTRLS_, "Action FC 2", train_, weightDecay_, gradAccumLength_, Xavier, true));
	if(temporalLength_ > 1){
		CUDAMallocZero(&temporalInput_, static_cast<size_t>(baseBatchSize_)*inC_*sizeof(__half));
		CUDAMallocZero(&temporalOut_, static_cast<size_t>(batchSize_)*outFeatureSize_*sizeof(__half));
		if(train_){
			CUDAMallocZero(&temporalGradReduced_, static_cast<size_t>(baseBatchSize_)*outFeatureSize_*sizeof(__half));
			CUDAMallocZero(&temporalGradOut_, static_cast<size_t>(batchSize_)*inC_*sizeof(__half));
		}
	}
}
ActionHead::~ActionHead(){
	for(const auto* layer : layers_) delete layer;
	layers_.clear();
	cudaFree(temporalInput_);
	cudaFree(temporalOut_);
	cudaFree(temporalGradReduced_);
	cudaFree(temporalGradOut_);
}
__half* ActionHead::Forward(__half* data){
	if(temporalLength_ > 1){
		SelectLastTemporalFrame(data, temporalInput_, baseBatchSize_, temporalLength_, inC_);
		data = temporalInput_;
	}
	for(auto* layer : layers_){
		data = layer->Forward(data);
	}
	if(temporalLength_ > 1){
		ExpandTemporalOutputs(data, temporalOut_, baseBatchSize_, temporalLength_, outFeatureSize_);
		return temporalOut_;
	}
	return data;
}
__half* ActionHead::Backward(__half* grad){
	if(temporalLength_ > 1){
		ReduceTemporalGradients(grad, temporalGradReduced_, baseBatchSize_, temporalLength_, outFeatureSize_);
		grad = temporalGradReduced_;
	}
	for(int i = static_cast<int>(layers_.size()); --i >= 0;){
		grad = layers_[i]->Backward(grad);
	}
	if(temporalLength_ > 1){
		ScatterLastTemporalFrameGrad(grad, temporalGradOut_, baseBatchSize_, temporalLength_, inC_);
		return temporalGradOut_;
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
