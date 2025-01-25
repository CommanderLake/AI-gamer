#include "ResFCLayer.h"
#include "Activate.h"
#include "common.h"
#include "BatchNorm.h"
#include "FCLayer.h"
ResFCLayer::ResFCLayer(cudaStream_t cudaStream, cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inC, int outC, const char* layerName, bool train, float weightDecay): cudaStream_(cudaStream), cudnnHandle_(cudnnHandle),
	batchSize_(batchSize){
	layerName_ = layerName;
	train_ = train;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC, 1, 1));
	layers_.push_back(new FCLayer(cudaStream, cudnnHandle, cublasHandle, batchSize_, inC, outC, "FC0", train, weightDecay));
	layers_.push_back(new BatchNorm(cudnnHandle_, CUDNN_BATCHNORM_SPATIAL, batchSize_, outC, 1, 1, "FC0 BatchNorm", train_, weightDecay));
	layers_.push_back(new Activate(cudnnHandle_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "FC0 ReLU"));
	layers_.push_back(new FCLayer(cudaStream, cudnnHandle_, cublasHandle, batchSize_, outC, outC, "FC1", train, weightDecay));
	layers_.push_back(new BatchNorm(cudnnHandle_, CUDNN_BATCHNORM_SPATIAL, batchSize_, outC, 1, 1, "FC1 BatchNorm", train_, weightDecay));
	residue_ = new FCLayer(cudaStream, cudnnHandle_, cublasHandle, batchSize_, inC, outC, "Residue", train, weightDecay);
	resAct_ = new Activate(cudnnHandle_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "Residue ReLU");
}
ResFCLayer::~ResFCLayer(){
	layers_.clear();
	delete residue_;
	delete resAct_;
}
__half* ResFCLayer::Forward(__half* data){
	const auto residue = residue_->Forward(data);
	for(int i = 0; i<layers_.size(); ++i){
		//std::cout << buttonLayers_[i]->layerName_ << " ";
		data = layers_[i]->Forward(data);
		//PrintDataHalf(buttonData, 14, "buttonData");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &blendFwd, residue_->outDesc_, residue, &blendFwd, layers_.back()->outDesc_, data));
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
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &blendBwd, inDesc_, residueGrad, &blendBwd, inDesc_, grad));
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
	return maxSize;
}
size_t ResFCLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(int i = 0; i<layers_.size(); ++i){
		maxSize = std::max(maxSize, layers_[i]->GetOptimizerStateSize());
	}
	return maxSize;
}