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
	tokensWithCls_ = nTokens_ + 1;
	CUDAMallocZero(&predictions_, batchSize_*NUM_CTRLS_*sizeof(__half));
	CUDAMallocZero(&classTokens_, static_cast<size_t>(batchSize_)*embedSize_*sizeof(__half));
	CUDAMallocZero(&patchTokens_, static_cast<size_t>(batchSize_)*nTokens_*embedSize_*sizeof(__half));
	CUDAMallocZero(&upstreamGrad_, static_cast<size_t>(batchSize_)*tokensWithCls_*embedSize_*sizeof(__half));
	int sharedH = patchRows_;
	int sharedW = patchCols_;
	trunkC1_ = RoundUp(embedSize_/2, 16);
	trunkC2_ = RoundUp(trunkC1_/2, 16);
	checkCUDNN(cudnnCreateTensorDescriptor(&sharedDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(sharedDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, trunkC2_, sharedHeight_, sharedWidth_));
	//sharedLayers_.push_back(new ViewerLayer(tokenElems, embedDim_, patchRows_, patchCols_, 16, "TokensToSpatial Viewer", true, 1.0f, false));
	sharedLayers_.push_back(new TokensToSpatialLayer(batchSize_, nTokens_, embedSize_, patchRows_, patchCols_, "TokensToSpatialLayer", train_));
	sharedLayers_.push_back(new ConvLayer(cudnn_, batchSize_, embedSize_, trunkC1_, 1, 1, 0, &sharedH, &sharedW, "Spatial Trunk Conv 1", train_, weightDecay_, gradAccumLength_, Xavier));
	sharedLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchSize_, trunkC1_, sharedH, sharedW, "Trunk BN 1", train_, gradAccumLength_));
	sharedLayers_.push_back(new GELULayer(batchSize_, trunkC1_, sharedH, sharedW, "Spatial Trunk GELU 1"));
	sharedLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, trunkC1_, sharedH, sharedW, "Trunk Drop 1", train_));
	//sharedLayers_.push_back(new ConvLayer(cudnn_, batchSize_, trunkC1_, trunkC2_, 1, 1, 0, &sharedH, &sharedW, "Spatial Trunk Conv 2", train_, weightDecay_, gradAccumLength_, Xavier));
	//sharedLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchSize_, trunkC2_, sharedH, sharedW, "Trunk BN 2", train_, gradAccumLength_));
	//sharedLayers_.push_back(new GELULayer(batchSize_, trunkC2_, sharedH, sharedW, "Spatial Trunk GELU 2"));
	//sharedLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, trunkC2_, sharedH, sharedW, "Trunk Drop 2", train_));
	//sharedLayers_.push_back(new ViewerLayer(tokenElems, embedDim_, patchRows_, patchCols_, 8, "Spatial Trunk Out Viewer", true, 1.0f, false));
	sharedHeight_ = sharedH;
	sharedWidth_ = sharedW;
	const int sharedSpatialSize = sharedHeight_*sharedWidth_;
	constexpr auto outC1 = 4096;
	neckDim_ = outC1;
	checkCUDNN(cudnnCreateTensorDescriptor(&neckDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(neckDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, neckDim_, 1, 1));
	CUDAMallocZero(&classNeckGrad_, static_cast<size_t>(batchSize_)*neckDim_*sizeof(__half));
	sharedLayers_.push_back(new FCLayer(cublas_, batchSize_, trunkC1_*sharedSpatialSize, outC1, "Trunk-neck FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	sharedLayers_.push_back(new LayerNorm(batchSize_, outC1, 1, 1, "Trunk-neck LN", train_));
	sharedLayers_.push_back(new GELULayer(batchSize_, outC1, 1, 1, "Trunk-neck GELU"));
	sharedLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, outC1, 1, 1, "Trunk-neck Drop", train_));
	classLayers_.push_back(new FCLayer(cublas_, batchSize_, embedSize_, outC1, "CLS FC", train_, weightDecay_, gradAccumLength_, Xavier, true));
	classLayers_.push_back(new LayerNorm(batchSize_, outC1, 1, 1, "CLS LN", train_));
	classLayers_.push_back(new GELULayer(batchSize_, outC1, 1, 1, "CLS GELU"));
	classLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, outC1, 1, 1, "CLS Drop", train_));
	constexpr auto outC2 = 1024;
	buttonLayers_.push_back(new FCLayer(cublas_, batchSize_, outC1, outC2, "Buttons FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	buttonLayers_.push_back(new LayerNorm(batchSize_, outC2, 1, 1, "Buttons LN", train_));
	buttonLayers_.push_back(new GELULayer(batchSize_, outC2, 1, 1, "Buttons GELU"));
	buttonLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, outC2, 1, 1, "Buttons Drop", train_));
	buttonLayers_.push_back(new FCLayer(cublas_, batchSize_, outC2, NUM_BUTS_, "Buttons FC 2", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new FCLayer(cublas_, batchSize_, outC1, outC2, "Axes FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new LayerNorm(batchSize_, outC2, 1, 1, "Axes LN", train_));
	axisLayers_.push_back(new GELULayer(batchSize_, outC2, 1, 1, "Axes GELU"));
	axisLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, outC2, 1, 1, "Axes Drop", train_));
	axisLayers_.push_back(new FCLayer(cublas_, batchSize_, outC2, NUM_AXES_, "Axes FC 2", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new AsinhLayer(batchSize_, NUM_AXES_, 1, 1, static_cast<int>(AXIS_SCALE_), "Axes Asinh"));
}
SpatialActionHead::~SpatialActionHead(){
	cudaFree(predictions_);
	cudaFree(upstreamGrad_);
	cudaFree(classNeckGrad_);
	cudaFree(patchTokens_);
	cudaFree(classTokens_);
	cudnnDestroyTensorDescriptor(sharedDesc_);
	cudnnDestroyTensorDescriptor(neckDesc_);
	for(const auto* layer : classLayers_) delete layer;
	for(const auto* layer : axisLayers_) delete layer;
	for(const auto* layer : buttonLayers_) delete layer;
	for(const auto* layer : sharedLayers_) delete layer;
	classLayers_.clear();
	axisLayers_.clear();
	buttonLayers_.clear();
	sharedLayers_.clear();
}
__half* SpatialActionHead::Forward(__half* data){
	StripClassToken(data, patchTokens_, batchSize_, embedSize_, nTokens_);
	auto tokenInput = patchTokens_;
	for(auto* layer : sharedLayers_){ tokenInput = layer->Forward(tokenInput); }
	GatherClassTokens(data, classTokens_, batchSize_, tokensWithCls_, embedSize_);
	auto classData = classTokens_;
	for(auto* layer : classLayers_){ classData = layer->Forward(classData); }
	checkCUDNN(cudnnAddTensor(cudnn_, &one_, neckDesc_, classData, &one_, neckDesc_, tokenInput));
	auto buttonData = tokenInput;
	for(auto* layer : buttonLayers_){ buttonData = layer->Forward(buttonData); }
	auto axisData = tokenInput;
	for(auto* layer : axisLayers_){ axisData = layer->Forward(axisData); }
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*batchSize_);
	return predictions_;
}
__half* SpatialActionHead::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad + NUM_BUTS_*batchSize_;
	for(int i = static_cast<int>(buttonLayers_.size()); --i >= 0;){ buttonGrad = buttonLayers_[i]->Backward(buttonGrad); }
	for(int i = static_cast<int>(axisLayers_.size()); --i >= 0;){ axisGrad = axisLayers_[i]->Backward(axisGrad); }
	checkCUDNN(cudnnAddTensor(cudnn_, &one_, sharedDesc_, axisGrad, &one_, sharedDesc_, buttonGrad));
	auto sharedGrad = buttonGrad;
	checkCUDA(cudaMemcpy(classNeckGrad_, sharedGrad, static_cast<size_t>(batchSize_)*neckDim_*sizeof(__half), cudaMemcpyDeviceToDevice));
	auto classGrad = classNeckGrad_;
	for(int i = static_cast<int>(classLayers_.size()); --i >= 0;){ classGrad = classLayers_[i]->Backward(classGrad); }
	for(int i = static_cast<int>(sharedLayers_.size()); --i >= 0;){ sharedGrad = sharedLayers_[i]->Backward(sharedGrad); }
	BuildClassTokenOutput(classGrad, sharedGrad, upstreamGrad_, batchSize_, embedSize_, nTokens_);
	return upstreamGrad_;
}
void SpatialActionHead::UpdateParameters(const float learningRate){
	for(auto* layer : sharedLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : buttonLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : axisLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : classLayers_){ layer->UpdateParameters(learningRate); }
}
void SpatialActionHead::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : classLayers_){ layer->SaveParameters(file, buffer); }
}
void SpatialActionHead::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : classLayers_){ layer->LoadParameters(file, buffer); }
}
void SpatialActionHead::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : classLayers_){ layer->SaveOptimizerState(file, buffer); }
}
void SpatialActionHead::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : classLayers_){ layer->LoadOptimizerState(file, buffer); }
}
size_t SpatialActionHead::GetParameterSize(){
	size_t maxSize = 0;
	for(auto* layer : sharedLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : classLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	return maxSize;
}
size_t SpatialActionHead::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(auto* layer : sharedLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : classLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	return maxSize;
}
void SpatialActionHead::SetTrain(const bool enable){
	train_ = enable;
	for(auto* layer : sharedLayers_){ layer->SetTrain(enable); }
	for(auto* layer : buttonLayers_){ layer->SetTrain(enable); }
	for(auto* layer : axisLayers_){ layer->SetTrain(enable); }
	for(auto* layer : classLayers_){ layer->SetTrain(enable); }
}