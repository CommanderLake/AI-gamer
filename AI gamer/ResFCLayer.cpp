#include "ResFCLayer.h"
#include "common.h"
#include "CuCommon.cuh"
#include "FCLayer.h"
#include "BatchNorm.h"
#include "GELULayer.h"
#include "LayerNorm.h"
ResFCLayer::ResFCLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int inC, const int hiddenC, const int outC, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength):
	cudnnHandle_(cudnnHandle), batchSize_(batchSize), inC_(inC), hiddenC_(hiddenC), outC_(outC), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, 1, 1));
	layers_.push_back(new FCLayer(cublasHandle, batchSize_, inC_, hiddenC_, "FC0", train, weightDecay, gradAccumLength_, Xavier));
	layers_.push_back(new LayerNorm(batchSize_, hiddenC_, 1, 1, "FC0 LayerNorm", train_));
	layers_.push_back(new GELULayer(batchSize_, hiddenC_, 1, 1, "GELU"));
	layers_.push_back(new FCLayer(cublasHandle, batchSize_, hiddenC_, outC_, "FC1", train, weightDecay, gradAccumLength_, Xavier));
	layers_.push_back(new LayerNorm(batchSize_, outC_, 1, 1, "FC1 LayerNorm", train_));
	residue_ = new FCLayer(cublasHandle, batchSize_, inC_, outC_, "Residue", train, weightDecay, gradAccumLength_, Xavier);
	resAct_ = new GELULayer(batchSize_, hiddenC_, 1, 1, "Residue GELU");
}
ResFCLayer::~ResFCLayer(){
	for(const auto layer : layers_) delete layer;
	layers_.clear();
	delete residue_;
	delete resAct_;
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(inDesc_));
}
__half* ResFCLayer::Forward(__half* data){
	const auto residue = residue_->Forward(data);
	for(int i = 0; i<layers_.size(); ++i){
		//std::cout << buttonLayers_[i]->layerName_ << " ";
		data = layers_[i]->Forward(data);
		//PrintDataHalf(buttonData, 14, "buttonData");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &fwdAlpha, outDesc_, residue, &fwdBeta, outDesc_, data));
	return resAct_->Forward(data);
}
__half* ResFCLayer::Backward(__half* grad){
	grad = resAct_->Backward(grad);
	const __half* residueGrad = residue_->Backward(grad);
	for(int i = layers_.size(); --i >= 0; ){
		//std::cout << buttonLayers_[i]->layerName_ << " ";
		grad = layers_[i]->Backward(grad);
		//PrintDataHalf(buttonGrad, 8, "gradient");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &bwdAlpha, inDesc_, residueGrad, &bwdBeta, inDesc_, grad));
	return grad;
}
void ResFCLayer::UpdateParameters(float learningRate){
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->UpdateParameters(learningRate);
	}
	residue_->UpdateParameters(learningRate);
}
void ResFCLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->SaveParameters(file, buffer);
	}
	residue_->SaveParameters(file, buffer);
}
void ResFCLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){ 
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->LoadParameters(file, buffer);
	}
	residue_->LoadParameters(file, buffer);
}
void ResFCLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){ 
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->SaveOptimizerState(file, buffer);
	}
	residue_->SaveOptimizerState(file, buffer);
}
void ResFCLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){ 
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->LoadOptimizerState(file, buffer);
	}
	residue_->LoadOptimizerState(file, buffer);
}
size_t ResFCLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(int i = 0; i<layers_.size(); ++i){
		maxSize = std::max(maxSize, layers_[i]->GetParameterSize());
	}
	maxSize = std::max(maxSize, residue_->GetParameterSize());
	return maxSize;
}
size_t ResFCLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(int i = 0; i<layers_.size(); ++i){
		maxSize = std::max(maxSize, layers_[i]->GetOptimizerStateSize());
	}
	maxSize = std::max(maxSize, residue_->GetOptimizerStateSize());
	return maxSize;
}
void ResFCLayer::SetTrain(bool enable){
	train_ = enable;
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->SetTrain(enable);
	}
	residue_->SetTrain(enable);
	resAct_->SetTrain(enable);
}