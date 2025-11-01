#include "SpatialActionHead.h"
#include "common.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "BatchNorm.h"
#include "GELULayer.h"
#include "Dropout.h"
#include "FCLayer.h"
#include "AsinhLayer.h"
#include "LayerNorm.h"
#include "TokensToSpatialLayer.h"
#include "ViewerLayer.h"
#undef min
#undef max
SpatialActionHead::SpatialActionHead(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int patchRows, const int patchCols, const int embedSize, const char* layerName, const bool train, const float weightDecay,
	const int gradAccumLength) : cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(batchSize), nTokens_(patchRows*patchCols), embedSize_(embedSize), patchRows_(patchRows), patchCols_(patchCols),
	sharedHeight_(patchRows), sharedWidth_(patchCols), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	CUDAMallocZero(&predictions_, batchSize_*NUM_CTRLS_*sizeof(__half));
	int sharedH = patchRows_;
	int sharedW = patchCols_;
	trunkC_ = RoundUp(embedSize_/4, 16);
	sharedLayers_.push_back(new TokensToSpatialLayer(batchSize_, nTokens_, embedSize_, patchRows_, patchCols_, "TokensToSpatialLayer", train_));
	sharedLayers_.push_back(new ConvLayer(cudnn_, batchSize_, embedSize_, trunkC_, 3, 1, 1, &sharedH, &sharedW, 1, "Spatial Conv", train_, weightDecay_, gradAccumLength_, Xavier));
	sharedLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchSize_, trunkC_, sharedH, sharedW, "Spatial BN", train_, gradAccumLength_));
	sharedLayers_.push_back(new GELULayer(batchSize_, trunkC_, sharedH, sharedW, "Spatial GELU"));
	sharedLayers_.push_back(new Dropout(cudnn_, 0.2f, batchSize_, trunkC_, sharedH, sharedW, "Spatial Drop", train_));
	sharedHeight_ = sharedH;
	sharedWidth_ = sharedW;
	sharedOutC_ = trunkC_*sharedHeight_*sharedWidth_;
	constexpr auto outC = 4096;
	checkCUDNN(cudnnCreateTensorDescriptor(&neckDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(neckDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC, 1, 1));
	sharedLayers_.push_back(new FCLayer(cublas_, batchSize_, sharedOutC_, outC, "Spatial-neck FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	sharedLayers_.push_back(new LayerNorm(batchSize_, outC, 1, 1, "Spatial-neck LN", train_));
	sharedLayers_.push_back(new GELULayer(batchSize_, outC, 1, 1, "Spatial-neck GELU"));
	sharedLayers_.push_back(new Dropout(cudnn_, 0.2f, batchSize_, outC, 1, 1, "Spatial-neck Drop", train_));
	buttonLayers_.push_back(new FCLayer(cublas_, batchSize_, outC, NUM_BUTS_, "Buttons FC 2", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new FCLayer(cublas_, batchSize_, outC, NUM_AXES_, "Axes FC 2", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new AsinhLayer(batchSize_, NUM_AXES_, 1, 1, static_cast<int>(AXIS_SCALE_), "Axes Asinh"));
}
SpatialActionHead::~SpatialActionHead(){
	cudaFree(predictions_);
	cudnnDestroyTensorDescriptor(neckDesc_);
	for(const auto* layer : axisLayers_) delete layer;
	for(const auto* layer : buttonLayers_) delete layer;
	for(const auto* layer : sharedLayers_) delete layer;
	axisLayers_.clear();
	buttonLayers_.clear();
	sharedLayers_.clear();
}
__half* SpatialActionHead::Forward(__half* data){
	for(auto* layer : sharedLayers_){ data = layer->Forward(data); }
	auto buttonData = data;
	for(auto* layer : buttonLayers_){ buttonData = layer->Forward(buttonData); }
	auto axisData = data;
	for(auto* layer : axisLayers_){ axisData = layer->Forward(axisData); }
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*batchSize_);
	return predictions_;
}
__half* SpatialActionHead::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad + NUM_BUTS_*batchSize_;
	for(int i = static_cast<int>(buttonLayers_.size()); --i >= 0;){ buttonGrad = buttonLayers_[i]->Backward(buttonGrad); }
	for(int i = static_cast<int>(axisLayers_.size()); --i >= 0;){ axisGrad = axisLayers_[i]->Backward(axisGrad); }
	checkCUDNN(cudnnAddTensor(cudnn_, &one_, neckDesc_, axisGrad, &one_, neckDesc_, buttonGrad));
	auto sharedGrad = buttonGrad;
	for(int i = static_cast<int>(sharedLayers_.size()); --i >= 0;){ sharedGrad = sharedLayers_[i]->Backward(sharedGrad); }
	return sharedGrad;
}
void SpatialActionHead::UpdateParameters(const float learningRate){
	for(auto* layer : sharedLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : buttonLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : axisLayers_){ layer->UpdateParameters(learningRate); }
}
void SpatialActionHead::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveParameters(file, buffer); }
}
void SpatialActionHead::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadParameters(file, buffer); }
}
void SpatialActionHead::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveOptimizerState(file, buffer); }
}
void SpatialActionHead::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadOptimizerState(file, buffer); }
}
size_t SpatialActionHead::GetParameterSize(){
	size_t maxSize = 0;
	for(auto* layer : sharedLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	return maxSize;
}
size_t SpatialActionHead::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(auto* layer : sharedLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	return maxSize;
}
void SpatialActionHead::SetTrain(const bool enable){
	train_ = enable;
	for(auto* layer : sharedLayers_){ layer->SetTrain(enable); }
	for(auto* layer : buttonLayers_){ layer->SetTrain(enable); }
	for(auto* layer : axisLayers_){ layer->SetTrain(enable); }
}