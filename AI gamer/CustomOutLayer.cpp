#include "CustomOutLayer.h"
#include "Activate.h"
#include "BatchNorm.h"
#include "common.h"
#include "Dropout.h"
#include "FCLayer.h"
#include "ResConvLayer.h"
#include "Sigmoid.h"
CustomOutLayer::CustomOutLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inputSize, const char* layerName, bool train, float weightDecay) : cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(batchSize), inC_(inputSize){
	layerName_ = layerName;
	train_ = train;
	cudaStreamCreate(&buttonStream_);
	cudaStreamCreate(&axisStream_);
	cudnnCreateTensorDescriptor(&inDesc_);
	cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inputSize, 1, 1);
	constexpr auto outC = 2048;
	buttonLayers_.push_back(new FCLayer(buttonStream_, cudnn_, cublas_, batchSize_, inputSize, outC, "Buts_FC1", train, weightDecay));
	buttonLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "Buts_FC1_BatchNorm", train, weightDecay));
	buttonLayers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "Buts_FC1_ReLU"));
	buttonLayers_.push_back(new Dropout(cudnn_, 0.5f, batchSize_, outC, 1, 1, "Buts_Dropout1", train));
	buttonLayers_.push_back(new FCLayer(buttonStream_, cudnn_, cublas_, batchSize_, outC, outC/2, "Buts_FC2", train, weightDecay));
	buttonLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC/2, 1, 1, "Buts_FC2_BatchNorm", train, weightDecay));
	buttonLayers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC/2, 1, 1, "Buts_FC2_ReLU"));
	buttonLayers_.push_back(new Dropout(cudnn_, 0.5f, batchSize_, outC/2, 1, 1, "Buts_Dropout2", train));
	buttonLayers_.push_back(new FCLayer(buttonStream_, cudnn_, cublas_, batchSize_, outC/2, NUM_BUTS_, "Buts_FC_Out", train, weightDecay));
	buttonLayers_.push_back(new Sigmoid(buttonStream_, NUM_BUTS_, batchSize_, NUM_BUTS_, "Buts_Sigmoid"));

	axisLayers_.push_back(new FCLayer(axisStream_, cudnn_, cublas_, batchSize_, inputSize, outC, "Axes_FC1", train, weightDecay));
	axisLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC, 1, 1, "Axes_FC1_BatchNorm", train, weightDecay));
	axisLayers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC, 1, 1, "Axes_FC1_ReLU"));
	axisLayers_.push_back(new Dropout(cudnn_, 0.5f, batchSize_, outC, 1, 1, "Axes_Dropout1", train));
	axisLayers_.push_back(new FCLayer(axisStream_, cudnn_, cublas_, batchSize_, outC, outC/2, "Axes_FC2", train, weightDecay));
	axisLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, outC/2, 1, 1, "Axes_FC2_BatchNorm", train, weightDecay));
	axisLayers_.push_back(new Activate(cudnn_, CUDNN_ACTIVATION_RELU, 1.0, batchSize_, outC/2, 1, 1, "Axes_FC2_ReLU"));
	axisLayers_.push_back(new Dropout(cudnn_, 0.5f, batchSize_, outC/2, 1, 1, "Axes_Dropout2", train));
	axisLayers_.push_back(new FCLayer(axisStream_, cudnn_, cublas_, batchSize_, outC/2, NUM_AXES_, "Axes_FC_Out", train, weightDecay));
	const auto outSizeBytes = (NUM_BUTS_+NUM_AXES_)*batchSize_*sizeof(__half);
	CUDAMallocZero(&outData_, outSizeBytes);
}
CustomOutLayer::~CustomOutLayer(){
	cudaFree(outData_);
	axisLayers_.clear();
	buttonLayers_.clear();
	cudnnDestroyTensorDescriptor(inDesc_);
	cudnnDestroyTensorDescriptor(outDesc_);
	cudaStreamDestroy(buttonStream_);
	cudaStreamDestroy(axisStream_);
}
__half* CustomOutLayer::Forward(__half* data){
	auto buttonData = data;
	auto axisData = data;
	cudaDeviceSynchronize();
	cublasSetStream(cublas_, buttonStream_);
	cudnnSetStream(cudnn_, buttonStream_);
	for(int i = 0; i<buttonLayers_.size(); ++i){
		//std::cout << "\n" << buttonLayers_[i]->layerName_ << " ";
		buttonData = buttonLayers_[i]->Forward(buttonData);
		//PrintDataHalf(buttonData, 16, "buttonData");
	}
	cublasSetStream(cublas_, axisStream_);
	cudnnSetStream(cudnn_, axisStream_);
	for(int i = 0; i<axisLayers_.size(); ++i){
		//std::cout << "\n" << axisLayers_[i]->layerName_ << " ";
		axisData = axisLayers_[i]->Forward(axisData);
		//PrintDataHalf(axisData, 16, "axisData");
	}
	cudaStreamSynchronize(buttonStream_);
	cudaStreamSynchronize(axisStream_);
	MergeOutputs(outData_, buttonData, axisData, NUM_CTRLS_*batchSize_, NUM_CTRLS_, NUM_BUTS_);
	return outData_;
}
__half* CustomOutLayer::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad + NUM_BUTS_*batchSize_;
	cudaDeviceSynchronize();
	cublasSetStream(cublas_, buttonStream_);
	cudnnSetStream(cudnn_, buttonStream_);
	for(int i = buttonLayers_.size(); --i >= 0; ){
		//std::cout << "\n" << buttonLayers_[i]->layerName_ << " ";
		buttonGrad = buttonLayers_[i]->Backward(buttonGrad);
		//PrintDataHalf(buttonGrad, 16, "buttonGrad");
	}
	cublasSetStream(cublas_, axisStream_);
	cudnnSetStream(cudnn_, axisStream_);
	for(int i = axisLayers_.size(); --i >= 0; ){
		//std::cout << "\n" << axisLayers_[i]->layerName_ << " ";
		axisGrad = axisLayers_[i]->Backward(axisGrad);
		//PrintDataHalf(axisGrad, 16, "axisGrad");
	}
	cudaStreamSynchronize(buttonStream_);
	cudaStreamSynchronize(axisStream_);
	checkCUDNN(cudnnAddTensor(cudnn_, &alpha, inDesc_, axisGrad, &alpha, inDesc_, buttonGrad));
	return buttonGrad;
}
void CustomOutLayer::UpdateParameters(float learningRate){
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->UpdateParameters(learningRate);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->UpdateParameters(learningRate);
	}
}
void CustomOutLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->SaveParameters(file, buffer);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->SaveParameters(file, buffer);
	}
}
void CustomOutLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->LoadParameters(file, buffer);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->LoadParameters(file, buffer);
	}
}
void CustomOutLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->SaveOptimizerState(file, buffer);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->SaveOptimizerState(file, buffer);
	}
}
void CustomOutLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->LoadOptimizerState(file, buffer);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->LoadOptimizerState(file, buffer);
	}
}
size_t CustomOutLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(int i = 0; i<buttonLayers_.size(); ++i){
		maxSize = std::max(maxSize, buttonLayers_[i]->GetParameterSize());
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		maxSize = std::max(maxSize, axisLayers_[i]->GetParameterSize());
	}
	return maxSize;
}
size_t CustomOutLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(int i = 0; i<buttonLayers_.size(); ++i){
		maxSize = std::max(maxSize, buttonLayers_[i]->GetOptimizerStateSize());
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		maxSize = std::max(maxSize, axisLayers_[i]->GetOptimizerStateSize());
	}
	return maxSize;
}
void CustomOutLayer::SetTrain(const bool enable){
	int bs;
	if(enable){
		train_ = true;
		bs = batchSize_;
	} else{
		train_ = false;
		bs = 1;
	}
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bs, inC_, 1, 1));
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->SetTrain(enable);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->SetTrain(enable);
	}
}
void CustomOutLayer::SetDropout(const bool enable){
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->SetDropout(enable);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->SetDropout(enable);
	}
}