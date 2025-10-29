#include "CustomOutLayer.h"
#include "common.h"
#include "CuCommon.cuh"
#include "FCLayer.h"
#include "GELULayer.h"
#include "AsinhLayer.h"
#include "Dropout.h"
#include "LayerNorm.h"
CustomOutLayer::CustomOutLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int seqLength, const int tokens, const int embedDim, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength) :
	cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(batchSize), seqLength_(seqLength), inC_(embedDim), tokens_(tokens), embedDim_(embedDim), fullFeatureSize_(tokens*embedDim), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	CUDAMallocZero(&upstreamGrad_, static_cast<size_t>(batchSize_)*fullFeatureSize_*sizeof(__half));
	CUDAMallocZero(&predictions_, batchSize_*NUM_CTRLS_*sizeof(__half));
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, 1, 1));
	const auto outC = 1024;
	buttonLayers_.push_back(new FCLayer(cublas_, batchSize_, inC_, outC, "Buttons FC 1", train_, weightDecay, gradAccumLength_, Xavier, true));
	buttonLayers_.push_back(new LayerNorm(batchSize_, outC, 1, 1, "Buttons LN", train_));
	buttonLayers_.push_back(new GELULayer(batchSize_, outC, 1, 1, "Buttons GELU"));
	buttonLayers_.push_back(new Dropout(cudnn_, 0.2f, batchSize_, outC, 1, 1, "Buttons Drop", train_));
	buttonLayers_.push_back(new FCLayer(cublas_, batchSize_, outC, NUM_BUTS_, "Buttons FC 2", train_, weightDecay, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new FCLayer(cublas_, batchSize_, inC_, outC, "Axes FC 1", train_, weightDecay, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new LayerNorm(batchSize_, outC, 1, 1, "Axes LN", train_));
	axisLayers_.push_back(new GELULayer(batchSize_, outC, 1, 1, "Axes GELU"));
	axisLayers_.push_back(new Dropout(cudnn_, 0.2f, batchSize_, outC, 1, 1, "Axes Drop", train_));
	axisLayers_.push_back(new FCLayer(cublas_, batchSize_, outC, NUM_AXES_, "Axes FC 2", train_, weightDecay, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new AsinhLayer(batchSize_, NUM_AXES_, 1, 1, static_cast<int>(AXIS_SCALE_), "Axes Asinh"));
}
CustomOutLayer::~CustomOutLayer(){
	cudaFree(predictions_);
	cudaFree(upstreamGrad_);
	cudnnDestroyTensorDescriptor(inDesc_);
	for(const auto layer : axisLayers_) delete layer;
	for(const auto layer : buttonLayers_) delete layer;
	axisLayers_.clear();
	buttonLayers_.clear();
}
__half* CustomOutLayer::Forward(__half* data){
	auto buttonData = data;
	for(auto* layer : buttonLayers_){ buttonData = layer->Forward(buttonData); }
	auto axisData = data;
	for(auto* layer : axisLayers_){ axisData = layer->Forward(axisData); }
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*batchSize_);
	return predictions_;
}
__half* CustomOutLayer::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad + NUM_BUTS_*batchSize_;
	for(int i = static_cast<int>(buttonLayers_.size()); --i >= 0;){ buttonGrad = buttonLayers_[i]->Backward(buttonGrad); }
	for(int i = static_cast<int>(axisLayers_.size()); --i >= 0;){ axisGrad = axisLayers_[i]->Backward(axisGrad); }
	checkCUDNN(cudnnAddTensor(cudnn_, &alpha_, inDesc_, axisGrad, &alpha_, inDesc_, buttonGrad));
	return upstreamGrad_;
}
void CustomOutLayer::UpdateParameters(const float learningRate){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->UpdateParameters(learningRate); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->UpdateParameters(learningRate); }
}
void CustomOutLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->SaveParameters(file, buffer); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->SaveParameters(file, buffer); }
}
void CustomOutLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->LoadParameters(file, buffer); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->LoadParameters(file, buffer); }
}
void CustomOutLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->SaveOptimizerState(file, buffer); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->SaveOptimizerState(file, buffer); }
}
void CustomOutLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->LoadOptimizerState(file, buffer); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->LoadOptimizerState(file, buffer); }
}
size_t CustomOutLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(int i = 0; i<buttonLayers_.size(); ++i){ maxSize = std::max(maxSize, buttonLayers_[i]->GetParameterSize()); }
	for(int i = 0; i<axisLayers_.size(); ++i){ maxSize = std::max(maxSize, axisLayers_[i]->GetParameterSize()); }
	return maxSize;
}
size_t CustomOutLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(int i = 0; i<buttonLayers_.size(); ++i){ maxSize = std::max(maxSize, buttonLayers_[i]->GetOptimizerStateSize()); }
	for(int i = 0; i<axisLayers_.size(); ++i){ maxSize = std::max(maxSize, axisLayers_[i]->GetOptimizerStateSize()); }
	return maxSize;
}
void CustomOutLayer::SetTrain(const bool enable){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->SetTrain(enable); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->SetTrain(enable); }
}