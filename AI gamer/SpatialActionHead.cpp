#include "SpatialActionHead.h"
#include "common.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "BatchNorm.h"
#include "GELULayer.h"
#include "Dropout.h"
#include "FCLayer.h"
#include "AsinhLayer.h"
#include "TokensToSpatialLayer.h"
#include "ViewerLayer.h"
#undef min
#undef max
SpatialActionHead::SpatialActionHead(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int seqLength, const int patchRows, const int patchCols, const int embedSize, const char* layerName, const bool train, const float weightDecay,
									const int gradAccumLength) : cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(batchSize), seqLength_(seqLength), nTokens_(patchRows*patchCols), embedSize_(embedSize), patchRows_(patchRows), patchCols_(patchCols),
																sharedHeight_(patchRows), sharedWidth_(patchCols), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	CUDAMallocZero(&spatialData_, static_cast<size_t>(batchSize_)*embedSize_*nTokens_*sizeof(__half));
	CUDAMallocZero(&tokenGrad_, static_cast<size_t>(batchSize_)*nTokens_*embedSize_*sizeof(__half));
	CUDAMallocZero(&predictions_, batchSize_*NUM_CTRLS_*sizeof(__half));
	int sharedH = patchRows_;
	int sharedW = patchCols_;
	trunkC1_ = RoundUp(embedSize_/2, 16);
	trunkC2_ = RoundUp(trunkC1_/2, 16);
	checkCUDNN(cudnnCreateTensorDescriptor(&sharedDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(sharedDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, trunkC2_, sharedHeight_, sharedWidth_));
	//sharedLayers_.push_back(new ViewerLayer(tokenElems, embedDim_, patchRows_, patchCols_, 16, "TokensToSpatial Viewer", true, 1.0f, false));
	sharedLayers_.push_back(new TokensToSpatialLayer(batchSize_, nTokens_, embedSize_, patchRows_, patchCols_, "TokensToSpatialLayer", train_));
	sharedLayers_.push_back(new ConvLayer(cudnn_, batchSize_, embedSize_, trunkC1_, 3, 1, 1, &sharedH, &sharedW, "Spatial Trunk Conv 1", train_, weightDecay_, gradAccumLength_, Xavier));
	sharedLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchSize_, trunkC1_, sharedH, sharedW, "Trunk BN 1", train_, gradAccumLength_));
	sharedLayers_.push_back(new GELULayer(batchSize_, trunkC1_, sharedH, sharedW, "Spatial Trunk GELU 1"));
	sharedLayers_.push_back(new Dropout(cudnn_, 0.2f, batchSize_, trunkC1_, sharedH, sharedW, "Trunk Drop 1", train_));
	sharedLayers_.push_back(new ConvLayer(cudnn_, batchSize_, trunkC1_, trunkC2_, 3, 1, 1, &sharedH, &sharedW, "Spatial Trunk Conv 2", train_, weightDecay_, gradAccumLength_, Xavier));
	sharedLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchSize_, trunkC2_, sharedH, sharedW, "Trunk BN 2", train_, gradAccumLength_));
	sharedLayers_.push_back(new GELULayer(batchSize_, trunkC2_, sharedH, sharedW, "Spatial Trunk GELU 2"));
	sharedLayers_.push_back(new Dropout(cudnn_, 0.2f, batchSize_, trunkC2_, sharedH, sharedW, "Trunk Drop 2", train_));
	//sharedLayers_.push_back(new ViewerLayer(tokenElems, embedDim_, patchRows_, patchCols_, 8, "Spatial Trunk Out Viewer", true, 1.0f, false));
	sharedHeight_ = sharedH;
	sharedWidth_ = sharedW;
	const int sharedSpatialSize = sharedHeight_*sharedWidth_;
	constexpr auto outC = 512;
	buttonLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, trunkC2_*sharedSpatialSize, outC, "Buttons FC 1", train_, weightDecay_, gradAccumLength_, Xavier, 1.0f, true));
	buttonLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "Buttons BN", train_, gradAccumLength_));
	buttonLayers_.push_back(new GELULayer(batchSize_, outC, 1, 1, "Buttons GELU"));
	buttonLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, outC, NUM_BUTS_, "Buttons FC 2", train_, weightDecay_, gradAccumLength_, Xavier, 1.0f, true));
	axisLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, trunkC2_*sharedSpatialSize, outC, "Axes FC 1", train_, weightDecay_, gradAccumLength_, Xavier, 1.0f, true));
	axisLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "Axes BN", train_, gradAccumLength_));
	axisLayers_.push_back(new GELULayer(batchSize_, outC, 1, 1, "Axes GELU"));
	axisLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_, outC, NUM_AXES_, "Axes FC 2", train_, weightDecay_, gradAccumLength_, Xavier, 1.0f, true));
	axisLayers_.push_back(new AsinhLayer(batchSize_, NUM_AXES_, 1, 1, static_cast<int>(AXIS_SCALE_), "Axes Asinh"));
}
SpatialActionHead::~SpatialActionHead(){
	cudaFree(spatialData_);
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
__half* SpatialActionHead::Forward(__half* data){
	auto spatialData = data;
	for(auto* layer : sharedLayers_){ spatialData = layer->Forward(spatialData); }
	auto buttonData = spatialData;
	for(auto* layer : buttonLayers_){ buttonData = layer->Forward(buttonData); }
	auto axisData = spatialData;
	for(auto* layer : axisLayers_){ axisData = layer->Forward(axisData); }
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*batchSize_);
	return predictions_;
}
__half* SpatialActionHead::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad + NUM_BUTS_*batchSize_;
	for(int i = static_cast<int>(buttonLayers_.size()); --i >= 0;){ buttonGrad = buttonLayers_[i]->Backward(buttonGrad); }
	for(int i = static_cast<int>(axisLayers_.size()); --i >= 0;){ axisGrad = axisLayers_[i]->Backward(axisGrad); }
	checkCUDNN(cudnnAddTensor(cudnn_, &alpha_, sharedDesc_, axisGrad, &alpha_, sharedDesc_, buttonGrad));
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
void SpatialActionHead::SetDropout(const bool enable){
	for(auto* layer : sharedLayers_){ layer->SetDropout(enable); }
	for(auto* layer : buttonLayers_){ layer->SetDropout(enable); }
	for(auto* layer : axisLayers_){ layer->SetDropout(enable); }
}