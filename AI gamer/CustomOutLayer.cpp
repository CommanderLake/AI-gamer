#include "CustomOutLayer.h"
#include "Activate.h"
#include "common.h"
#include "FCLayer.h"
#include "ResConvLayer.h"
#include "ResFCLayer.h"
#include "Sigmoid.h"
CustomOutLayer::CustomOutLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inputSize, const char* layerName, bool train, float weightDecay) : cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(batchSize){
	layerName_ = layerName;
	train_ = train;
	cudaStreamCreate(&buttonStream_);
	cudaStreamCreate(&axisStream_);
	cudnnCreateTensorDescriptor(&outDesc_);
	cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, numCtrls_, 1, 1);
	cudnnCreateTensorDescriptor(&inDesc_);
	cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inputSize, 1, 1);
	auto outC = 512;
	buttonLayers_.push_back(new ResFCLayer(buttonStream_, cudnn_, cublas_, batchSize_, inputSize, outC, "Binary_ResFC1", train, weightDecay));
	buttonLayers_.push_back(new ResFCLayer(buttonStream_, cudnn_, cublas_, batchSize_, outC, outC, "Binary_ResFC2", train, weightDecay));
	buttonLayers_.push_back(new ResFCLayer(buttonStream_, cudnn_, cublas_, batchSize_, outC, outC, "Binary_ResFC3", train, weightDecay));
	buttonLayers_.push_back(new ResFCLayer(buttonStream_, cudnn_, cublas_, batchSize_, outC, outC, "Binary_ResFC4", train, weightDecay));
	outC = numButs_;
	buttonLayers_.push_back(new FCLayer(buttonStream_, cudnn_, cublas_, batchSize_, 512, outC, "Binary_FC_Out", train, weightDecay));
	buttonLayers_.push_back(new Sigmoid(buttonStream_, numButs_, batchSize_, outC, "Binary_Sigmoid"));

	outC = 512;
	axisLayers_.push_back(new ResFCLayer(axisStream_, cudnn_, cublas_, batchSize_, inputSize, outC, "Continuous_ResFC1", train, weightDecay));
	axisLayers_.push_back(new ResFCLayer(axisStream_, cudnn_, cublas_, batchSize_, outC, outC, "Continuous_ResFC2", train, weightDecay));
	axisLayers_.push_back(new ResFCLayer(axisStream_, cudnn_, cublas_, batchSize_, outC, outC, "Continuous_ResFC3", train, weightDecay));
	axisLayers_.push_back(new ResFCLayer(axisStream_, cudnn_, cublas_, batchSize_, outC, outC, "Continuous_ResFC4", train, weightDecay));
	outC = numAxes_;
	axisLayers_.push_back(new FCLayer(axisStream_, cudnn_, cublas_, batchSize_, 512, outC, "Continuous_FC_Out", train, weightDecay));
	const auto outSizeBytes = (numButs_+numAxes_)*batchSize_*sizeof(__half);
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
		//std::cout << "\r\n" << buttonLayers_[i]->layerName_ << " ";
		buttonData = buttonLayers_[i]->Forward(buttonData);
		//PrintDataHalf(buttonData, 14, "buttonData");
	}
	cublasSetStream(cublas_, axisStream_);
	cudnnSetStream(cudnn_, axisStream_);
	for(int i = 0; i<axisLayers_.size(); ++i){
		//std::cout << "\r\n" << axisLayers_[i]->layerName_ << " ";
		axisData = axisLayers_[i]->Forward(axisData);
		//PrintDataHalf(axisData, 8, "axisData");
	}
	cudaStreamSynchronize(buttonStream_);
	cudaStreamSynchronize(axisStream_);
	MergeOutputs(outData_, buttonData, axisData, numCtrls_*batchSize_, numCtrls_, numButs_);
	return outData_;
}
__half* CustomOutLayer::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad + numButs_*batchSize_;
	cudaDeviceSynchronize();
	cublasSetStream(cublas_, buttonStream_);
	cudnnSetStream(cudnn_, buttonStream_);
	for(int i = buttonLayers_.size(); --i >= 0; ){
		//std::cout << "\r\n" << buttonLayers_[i]->layerName_ << " ";
		buttonGrad = buttonLayers_[i]->Backward(buttonGrad);
		//PrintDataHalf(buttonGrad, 8, "buttonGrad");
	}
	cublasSetStream(cublas_, axisStream_);
	cudnnSetStream(cudnn_, axisStream_);
	for(int i = axisLayers_.size(); --i >= 0; ){
		//std::cout << "\r\n" << axisLayers_[i]->layerName_ << " ";
		axisGrad = axisLayers_[i]->Backward(axisGrad);
		//PrintDataHalf(axisGrad, 8, "axisGrad");
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