#include "ResConvLayer.h"
#include "Activate.h"
#include "common.h"
#include "BatchNorm.h"
#include "ConvLayer.h"
ResConvLayer::ResConvLayer(cudnnHandle_t cudnnHandle, int batchSize, int inC, int outC, int *inHeight, int *inWidth, const char* layerName, bool train, float weightDecay): cudnnHandle_(cudnnHandle), batchSize_(batchSize), outC_(outC), inC_(inC),
	inHeight_(*inHeight), inWidth_(*inWidth){
	layerName_ = layerName;
	train_ = train;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, *inHeight, *inWidth));
	int resH = *inHeight, resW = *inWidth;
	layers_.push_back(new ConvLayer(cudnnHandle_, batchSize_, inC_, outC_, 4, 2, 1, inHeight, inWidth, "Conv0", train, weightDecay));
	layers_.push_back(new BatchNorm(cudnnHandle_, CUDNN_BATCHNORM_SPATIAL, batchSize_, outC_, *inHeight, *inWidth, "Conv0 BatchNorm", train_, weightDecay));
	layers_.push_back(new Activate(cudnnHandle_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC_, *inHeight, *inWidth, "Conv0 ReLU"));
	layers_.push_back(new ConvLayer(cudnnHandle_, batchSize_, outC_, outC_, 3, 1, 1, inHeight, inWidth, "Conv1", train, weightDecay));
	layers_.push_back(new BatchNorm(cudnnHandle_, CUDNN_BATCHNORM_SPATIAL, batchSize_, outC_, *inHeight, *inWidth, "Conv1 BatchNorm", train_, weightDecay));
	residue_ = new ConvLayer(cudnnHandle_, batchSize_, inC_, outC_, 2, 2, 0, &resH, &resW, "Residue", train, weightDecay);
	resAct_ = new Activate(cudnnHandle_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC_, *inHeight, *inWidth, "Residue ReLU");
	outWidth_ = *inWidth;
	outHeight_ = *inHeight;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, outHeight_, outWidth_));
}
ResConvLayer::~ResConvLayer(){
	layers_.clear();
	delete residue_;
	delete resAct_;
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(inDesc_));
}
__half* ResConvLayer::Forward(__half* data){
	const auto residue = residue_->Forward(data);
	for(int i = 0; i<layers_.size(); ++i){
		//std::cout << layers_[i]->layerName_ << " ";
		data = layers_[i]->Forward(data);
		//PrintDataHalf(data, 14, "data");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &blendFwd, residue_->outDesc_, residue, &blendFwd, layers_.back()->outDesc_, data));
	return resAct_->Forward(data);
}
__half* ResConvLayer::Backward(__half* grad){
	grad = resAct_->Backward(grad);
	const __half* residueGrad = residue_->Backward(grad);
	for(int i = layers_.size(); --i >= 0; ){
		//std::cout << layers_[i]->layerName_ << " ";
		grad = layers_[i]->Backward(grad);
		//PrintDataHalf(grad, 8, "gradient");
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &blendBwd, inDesc_, residueGrad, &blendBwd, inDesc_, grad));
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
	maxSize = std::max(maxSize, residue_->GetParameterSize());
	return maxSize;
}
size_t ResConvLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(int i = 0; i<layers_.size(); ++i){
		maxSize = std::max(maxSize, layers_[i]->GetOptimizerStateSize());
	}
	maxSize = std::max(maxSize, residue_->GetOptimizerStateSize());
	return maxSize;
}
void ResConvLayer::SetTrain(bool enable){
	int bs;
	if(enable){
		train_ = true;
		bs = batchSize_;
	} else{
		train_ = false;
		bs = 1;
	}
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, inC_, inHeight_, inWidth_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, outC_, outHeight_, outWidth_));
	for(int i = 0; i<layers_.size(); ++i){
		layers_[i]->SetTrain(enable);
	}
	residue_->SetTrain(enable);
	resAct_->SetTrain(enable);
}