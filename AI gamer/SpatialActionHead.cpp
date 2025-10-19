#include "SpatialActionHead.h"
#include "common.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "GELULayer.h"
#include "PoolLayer.h"
#include "FCLayer.h"
#include "SigmoidLayer.h"
#include "ViewerLayer.h"
#undef min
#undef max
namespace{
	constexpr int kSharedChannels1 = 128;
	constexpr int kSharedChannels2 = 64;
	constexpr int kAxisHiddenDim = 128;
}
SpatialActionHead::SpatialActionHead(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int seqLength, const int patchRows, const int patchCols, const int embedDim, const char* layerName, const bool train, const float weightDecay,
									const int gradAccumLength) : cudnn_(cudnnHandle), cublas_(cublasHandle), ogbs_(batchSize), batchSize_(batchSize), seqLength_(seqLength), nTokens_(patchRows*patchCols), patchRows_(patchRows), patchCols_(patchCols), embedDim_(embedDim), sharedChannels_(kSharedChannels2),
																sharedHeight_(patchRows), sharedWidth_(patchCols), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	const auto spatialElems = static_cast<size_t>(batchSize_)*embedDim_*nTokens_;
	const auto tokenElems = static_cast<size_t>(batchSize_)*nTokens_*embedDim_;
	CUDAMallocZero(&spatialInput_, spatialElems*sizeof(__half));
	CUDAMallocZero(&tokenGrad_, tokenElems*sizeof(__half));
	CUDAMallocZero(&predictions_, batchSize_*NUM_CTRLS_*sizeof(__half));
	int sharedH = patchRows_;
	int sharedW = patchCols_;
	sharedLayers_.push_back(new ViewerLayer(embedDim_, patchRows_, patchCols_, 16, "Spatial_Trunk_In_Viewer"));
	sharedLayers_.push_back(new ConvLayer(cudnn_, batchSize_, embedDim_, kSharedChannels1, 1, 1, &sharedH, &sharedW, "Spatial_Trunk_Conv1", train_, weightDecay_, gradAccumLength_, Xavier));
	sharedLayers_.push_back(new GELULayer(batchSize_, kSharedChannels1, sharedH, sharedW, "Spatial_Trunk_GELU1"));
	sharedLayers_.push_back(new ConvLayer(cudnn_, batchSize_, kSharedChannels1, kSharedChannels2, 1, 1, &sharedH, &sharedW, "Spatial_Trunk_Conv2", train_, weightDecay_, gradAccumLength_, Xavier));
	sharedLayers_.push_back(new GELULayer(batchSize_, kSharedChannels2, sharedH, sharedW, "Spatial_Trunk_GELU2"));
	sharedLayers_.push_back(new ViewerLayer(kSharedChannels2, patchRows_, patchCols_, 8, "Spatial_Trunk_Out_Viewer"));
	sharedChannels_ = kSharedChannels2;
	sharedHeight_ = sharedH;
	sharedWidth_ = sharedW;
	int buttonH = sharedHeight_;
	int buttonW = sharedWidth_;
	buttonLayers_.push_back(new PoolLayer(cudnn_, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, batchSize_, sharedChannels_, &buttonH, &buttonW, buttonH, buttonW, buttonH, buttonW, "Buttons_GlobalPool", train_));
	buttonLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, sharedChannels_, NUM_BUTS_, "Buttons_FC", train_, weightDecay_, gradAccumLength_, Xavier, 1.0f, true));
	buttonLayers_.push_back(new SigmoidLayer(batchSize_, NUM_BUTS_, NUM_BUTS_, "Buttons_Sigmoid"));
	int axisH = sharedHeight_;
	int axisW = sharedWidth_;
	axisLayers_.push_back(new PoolLayer(cudnn_, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, batchSize_, sharedChannels_, &axisH, &axisW, axisH, axisW, axisH, axisW, "Axes_GlobalPool", train_));
	axisLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, sharedChannels_, kAxisHiddenDim, "Axes_FC1", train_, weightDecay_, gradAccumLength_, Xavier, 1.0f, true));
	axisLayers_.push_back(new GELULayer(batchSize_, kAxisHiddenDim, 1, 1, "Axes_GELU"));
	axisLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, kAxisHiddenDim, NUM_AXES_, "Axes_FC2", train_, weightDecay_, gradAccumLength_, Xavier, 1.0f, true));
	checkCUDNN(cudnnCreateTensorDescriptor(&sharedDesc_));
	UpdateSharedDescriptor();
}
SpatialActionHead::~SpatialActionHead(){
	cudaFree(spatialInput_);
	cudaFree(tokenGrad_);
	cudaFree(predictions_);
	cudnnDestroyTensorDescriptor(sharedDesc_);
	for(auto* layer : axisLayers_){ delete layer; }
	for(auto* layer : buttonLayers_){ delete layer; }
	for(auto* layer : sharedLayers_){ delete layer; }
	axisLayers_.clear();
	buttonLayers_.clear();
	sharedLayers_.clear();
}
void SpatialActionHead::UpdateSharedDescriptor(){
	checkCUDNN(cudnnSetTensor4dDescriptor(sharedDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, sharedChannels_, sharedHeight_, sharedWidth_));
}
__half* SpatialActionHead::Forward(__half* data){
	const int effectiveBatch = batchSize_;
	TokensToSpatial(data, spatialInput_, effectiveBatch, nTokens_, embedDim_, patchRows_, patchCols_);
	auto spatialData = spatialInput_;
	for(auto* layer : sharedLayers_){ spatialData = layer->Forward(spatialData); }
	sharedOutput_ = spatialData;
	auto buttonData = sharedOutput_;
	for(auto* layer : buttonLayers_){ buttonData = layer->Forward(buttonData); }
	auto axisData = sharedOutput_;
	for(auto* layer : axisLayers_){ axisData = layer->Forward(axisData); }
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*effectiveBatch);
	return predictions_;
}
__half* SpatialActionHead::Backward(__half* grad){
	const int effectiveBatch = batchSize_;
	auto buttonGrad = grad;
	auto axisGrad = grad + NUM_BUTS_*effectiveBatch;
	for(int i = static_cast<int>(buttonLayers_.size()); --i >= 0;){ buttonGrad = buttonLayers_[i]->Backward(buttonGrad); }
	for(int i = static_cast<int>(axisLayers_.size()); --i >= 0;){ axisGrad = axisLayers_[i]->Backward(axisGrad); }
	checkCUDNN(cudnnAddTensor(cudnn_, &alpha_, sharedDesc_, axisGrad, &alpha_, sharedDesc_, buttonGrad));
	auto sharedGrad = buttonGrad;
	for(int i = static_cast<int>(sharedLayers_.size()); --i >= 0;){ sharedGrad = sharedLayers_[i]->Backward(sharedGrad); }
	SpatialToTokens(sharedGrad, tokenGrad_, effectiveBatch, nTokens_, embedDim_, patchRows_, patchCols_);
	return tokenGrad_;
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
	batchSize_ = enable ? ogbs_ : 1;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	UpdateSharedDescriptor();
	for(auto* layer : sharedLayers_){ layer->SetTrain(enable); }
	for(auto* layer : buttonLayers_){ layer->SetTrain(enable); }
	for(auto* layer : axisLayers_){ layer->SetTrain(enable); }
}
void SpatialActionHead::SetDropout(const bool enable){
	for(auto* layer : sharedLayers_){ layer->SetDropout(enable); }
	for(auto* layer : buttonLayers_){ layer->SetDropout(enable); }
	for(auto* layer : axisLayers_){ layer->SetDropout(enable); }
}