#include "CustomOutLayer.h"
#include "Activate.h"
#include "BatchNorm.h"
#include "common.h"
#include "Dropout.h"
#include "FCLayer.h"
#include "ResConvLayer.h"
#include "SigmoidLayer.h"
#include "ViewerLayer.h"
CustomOutLayer::CustomOutLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int seqLength, const int inputSize, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength) :
	cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(batchSize), seqLength_(seqLength), inC_(inputSize), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	cudnnCreateTensorDescriptor(&inDesc_);
	cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_*seqLength_, inputSize, 1, 1);
	constexpr auto outC = 2048;
	//buttonLayers_.push_back(new LSTMLayer(cudnn_, seqLength_, 1, outC, batchSize_, inputSize, "Buts_LSTM", train, weightDecay, gradAccumLength_));
	//buttonLayers_.push_back(new ViewerLayer(batchSize_*seqLength_, 32, 32, 10, "Buts LSTM"));
	buttonLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_*seqLength_, inputSize, outC, "Buts_FC1", train, weightDecay, gradAccumLength_));
	buttonLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_*seqLength_, outC, 1, 1, "Buts_FC1_BatchNorm", train, weightDecay, gradAccumLength_));
	buttonLayers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_*seqLength_, outC, 1, 1, "Buts_FC1_ReLU"));
	buttonLayers_.push_back(new Dropout(cudnn_, 0.5f, batchSize_*seqLength_, outC, 1, 1, "Buts_Dropout1", train));
	buttonLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_*seqLength_, outC, outC/2, "Buts_FC2", train, weightDecay, gradAccumLength_));
	buttonLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_*seqLength_, outC/2, 1, 1, "Buts_FC2_BatchNorm", train, weightDecay, gradAccumLength_));
	buttonLayers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_*seqLength_, outC/2, 1, 1, "Buts_FC2_ReLU"));
	buttonLayers_.push_back(new Dropout(cudnn_, 0.5f, batchSize_*seqLength_, outC/2, 1, 1, "Buts_Dropout2", train));
	buttonLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_*seqLength_, outC/2, NUM_BUTS_, "Buts_FC_Out", train, weightDecay, gradAccumLength_));
	buttonLayers_.push_back(new SigmoidLayer(NUM_BUTS_, batchSize_*seqLength_, NUM_BUTS_, "Buts_Sigmoid"));

	//axisLayers_.push_back(new LSTMLayer(cudnn_, seqLength_, 1, outC, batchSize_, inputSize, "Axes_LSTM", train, weightDecay, gradAccumLength_));
	//axisLayers_.push_back(new ViewerLayer(batchSize_*seqLength_, 32, 32, 10, "Axes LSTM"));
	axisLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_*seqLength_, inputSize, outC, "Axes_FC1", train, weightDecay, gradAccumLength_));
	axisLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_*seqLength_, outC, 1, 1, "Axes_FC1_BatchNorm", train, weightDecay, gradAccumLength_));
	axisLayers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_*seqLength_, outC, 1, 1, "Axes_FC1_ReLU"));
	axisLayers_.push_back(new Dropout(cudnn_, 0.5f, batchSize_*seqLength_, outC, 1, 1, "Axes_Dropout1", train));
	axisLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_*seqLength_, outC, outC/2, "Axes_FC2", train, weightDecay, gradAccumLength_));
	axisLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_*seqLength_, outC/2, 1, 1, "Axes_FC2_BatchNorm", train, weightDecay, gradAccumLength_));
	axisLayers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_*seqLength_, outC/2, 1, 1, "Axes_FC2_ReLU"));
	axisLayers_.push_back(new Dropout(cudnn_, 0.5f, batchSize_*seqLength_, outC/2, 1, 1, "Axes_Dropout2", train));
	axisLayers_.push_back(new FCLayer(cudnn_, cublas_, batchSize_*seqLength_, outC/2, NUM_AXES_, "Axes_FC_Out", train, weightDecay, gradAccumLength_));
	CUDAMallocZero(&predictions_, batchSize_*seqLength_*NUM_CTRLS_*sizeof(__half));
}
CustomOutLayer::~CustomOutLayer(){
	cudaFree(predictions_);
	axisLayers_.clear();
	buttonLayers_.clear();
	cudnnDestroyTensorDescriptor(inDesc_);
	cudnnDestroyTensorDescriptor(outDesc_);
}
__half* CustomOutLayer::Forward(__half* data){
	auto buttonData = data;
	auto axisData = data;
	for(int i = 0; i<buttonLayers_.size(); ++i){
		//std::cout << "\n" << buttonLayers_[i]->layerName_ << " ";
		buttonData = buttonLayers_[i]->Forward(buttonData);
		//PrintDataHalf(buttonData, 16, "buttonData");
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		//std::cout << "\n" << axisLayers_[i]->layerName_ << " ";
		axisData = axisLayers_[i]->Forward(axisData);
		//PrintDataHalf(axisData, 16, "axisData");
	}
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*batchSize_*seqLength_);
	return predictions_;
}
__half* CustomOutLayer::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad+NUM_BUTS_*batchSize_*seqLength_;
	for(int i = buttonLayers_.size(); --i>=0;){
		//std::cout << "\n" << buttonLayers_[i]->layerName_ << " ";
		buttonGrad = buttonLayers_[i]->Backward(buttonGrad);
		//PrintDataHalf(buttonGrad, 16, "buttonGrad");
	}
	for(int i = axisLayers_.size(); --i>=0;){
		//std::cout << "\n" << axisLayers_[i]->layerName_ << " ";
		axisGrad = axisLayers_[i]->Backward(axisGrad);
		//PrintDataHalf(axisGrad, 16, "axisGrad");
	}
	checkCUDNN(cudnnAddTensor(cudnn_, &alpha, inDesc_, axisGrad, &alpha, inDesc_, buttonGrad));
	return buttonGrad;
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
	for(int i = 0; i<buttonLayers_.size(); ++i){ maxSize = max(maxSize, buttonLayers_[i]->GetParameterSize()); }
	for(int i = 0; i<axisLayers_.size(); ++i){ maxSize = max(maxSize, axisLayers_[i]->GetParameterSize()); }
	return maxSize;
}
size_t CustomOutLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(int i = 0; i<buttonLayers_.size(); ++i){ maxSize = max(maxSize, buttonLayers_[i]->GetOptimizerStateSize()); }
	for(int i = 0; i<axisLayers_.size(); ++i){ maxSize = max(maxSize, axisLayers_[i]->GetOptimizerStateSize()); }
	return maxSize;
}
void CustomOutLayer::SetTrain(const bool enable){
	int bs;
	if(enable){
		train_ = true;
		bs = batchSize_;
		batchSize_ = batchSize_*seqLength_;
	} else{
		train_ = false;
		bs = 1;
		batchSize_ = batchSize_;
	}
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, inC_, 1, 1));
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->SetTrain(enable); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->SetTrain(enable); }
}
void CustomOutLayer::SetDropout(const bool enable){
	for(int i = 0; i<buttonLayers_.size(); ++i){ buttonLayers_[i]->SetDropout(enable); }
	for(int i = 0; i<axisLayers_.size(); ++i){ axisLayers_[i]->SetDropout(enable); }
}