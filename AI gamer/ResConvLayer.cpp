#include "ResConvLayer.h"
#include "Activate.h"
#include "common.h"
#include "BatchNorm.h"
#include "ConvLayer.h"
#include "LeakyReLU.h"
#include "Swish.h"
ResConvLayer::ResConvLayer(cudnnHandle_t cudnnHandle, int batchSize, int inC, int outC, int *inHeight, int *inWidth, const char* layerName, bool train): cudnnHandle_(cudnnHandle), batchSize_(batchSize){
	layerName_ = layerName;
	train_ = train;
	constexpr auto wd = 0.000001f;
	int resH = *inHeight, resW = *inWidth;
	layers_.push_back(new ConvLayer(cudnnHandle_, batchSize_, inC, outC, 4, 2, 1, inHeight, inWidth, "Conv0", train, wd));
	layers_.push_back(new BatchNorm(cudnnHandle_, CUDNN_BATCHNORM_SPATIAL, batchSize_, outC, *inHeight, *inWidth, "Conv0 BatchNorm", train_, wd));
	//layers_.push_back(new Swish(batchSize_*outC**inHeight**inWidth, "Conv0 Swish"));
	layers_.push_back(new Activate(cudnnHandle_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, *inHeight, *inWidth, "Conv0 ReLU"));
	layers_.push_back(new ConvLayer(cudnnHandle_, batchSize_, outC, outC, 3, 1, 1, inHeight, inWidth, "Conv1", train, wd));
	layers_.push_back(new BatchNorm(cudnnHandle_, CUDNN_BATCHNORM_SPATIAL, batchSize_, outC, *inHeight, *inWidth, "Conv1 BatchNorm", train_, wd));
	residue_ = new ConvLayer(cudnnHandle_, batchSize_, inC, outC, 2, 2, 0, &resH, &resW, "Residue", train, wd);
	//resAct_ = new Swish(batchSize_*outC*resH*resW, "Residue Swish");
	resAct_ = new Activate(cudnnHandle_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, *inHeight, *inWidth, "Residue ReLU");
}
ResConvLayer::~ResConvLayer(){
	layers_.clear();
	delete residue_;
	delete resAct_;
}
__half* ResConvLayer::Forward(__half* data){
	const auto residue = residue_->Forward(data);
	for(int i = 0; i<layers_.size(); ++i){
		//std::cout << buttonLayers_[i]->layerName_ << " ";
		data = layers_[i]->Forward(data);
		//PrintDataHalf(buttonData, 14, "buttonData");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, residue_->outDesc_, residue, &alpha, layers_.back()->outDesc_, data));
	return resAct_->Forward(data);
}
__half* ResConvLayer::Backward(__half* grad){
	grad = resAct_->Backward(grad);
	const __half* residueGrad = residue_->Backward(grad);
	for(int i = layers_.size(); --i >= 0; ){
		//std::cout << buttonLayers_[i]->layerName_ << " ";
		grad = layers_[i]->Backward(grad);
		//PrintDataHalf(buttonGrad, 8, "gradient");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, ((ConvLayer*)layers_[0])->inDesc_, residueGrad, &alpha, ((ConvLayer*)layers_[0])->inDesc_, grad));
	return grad;
}
void ResConvLayer::UpdateParameters(float learningRate){
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->UpdateParameters(learningRate);
	}
	residue_->UpdateParameters(learningRate);
}
void ResConvLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->SaveParameters(file, buffer);
	}
	residue_->SaveParameters(file, buffer);
}
void ResConvLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){ 
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->LoadParameters(file, buffer);
	}
	residue_->LoadParameters(file, buffer);
}
void ResConvLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){ 
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->SaveOptimizerState(file, buffer);
	}
	residue_->SaveOptimizerState(file, buffer);
}
void ResConvLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){ 
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->LoadOptimizerState(file, buffer);
	}
	residue_->LoadOptimizerState(file, buffer);
}
size_t ResConvLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(int i = 0; i<layers_.size(); ++i){
		maxSize = std::max(maxSize, layers_[i]->GetParameterSize());
	}
	return maxSize;
}
size_t ResConvLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(int i = 0; i<layers_.size(); ++i){
		maxSize = std::max(maxSize, layers_[i]->GetOptimizerStateSize());
	}
	return maxSize;
}