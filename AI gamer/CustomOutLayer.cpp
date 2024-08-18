#include "CustomOutLayer.h"
#include "common.h"
#include "FCLayer.h"
#include "BatchNorm.h"
#include "Dropout.h"
#include "LeakyReLU.h"
#include "Sigmoid.h"
CustomOutLayer::CustomOutLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, cudnnTensorDescriptor_t outDesc, int batchSize, int inputSize, int hiddenSize, int buttons, int axes, const char* layerName, bool train): cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle),
	batchSize_(batchSize), buttons_(buttons), axes_(axes){
	layerName_ = layerName;
	train_ = train;
	outDesc_ = outDesc;
	constexpr auto wdButs = 0.001f;
	constexpr auto wdaxes = 0.001f;
	buttonLayers_.push_back(new FCLayer(cudnnHandle_, cublasHandle_, batchSize_, inputSize, hiddenSize, "Buttons FC 0", train_, wdButs));
	buttonLayers_.push_back(new BatchNorm(cudnnHandle_, buttonLayers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "Buttons BN 0", train_, wdButs));
	buttonLayers_.push_back(new LeakyReLU(buttonLayers_.back()->outDesc_, "Buttons Leaky ReLU 0"));
	buttonLayers_.push_back(new Dropout(cudnnHandle_, buttonLayers_.back()->outDesc_, 0.2f, "Buttons Drop 0"));
	buttonLayers_.push_back(new FCLayer(cudnnHandle_, cublasHandle_, batchSize_, hiddenSize, buttons, "Buttons FC 1", train_, wdButs));
	buttonLayers_.push_back(new Sigmoid(buttonLayers_.back()->outDesc_, buttons, batchSize_, "Buttons Sigmoid 0"));
	axisLayers_.push_back(new FCLayer(cudnnHandle_, cublasHandle_, batchSize_, inputSize, hiddenSize, "Axes FC 0", train_, wdaxes));
	axisLayers_.push_back(new BatchNorm(cudnnHandle_, axisLayers_.back()->outDesc_, CUDNN_BATCHNORM_PER_ACTIVATION, batchSize_, "Axes BN 0", train_, wdaxes));
	axisLayers_.push_back(new LeakyReLU(axisLayers_.back()->outDesc_, "Axes Leaky ReLU 0"));
	axisLayers_.push_back(new Dropout(cudnnHandle_, axisLayers_.back()->outDesc_, 0.2f, "Axes Drop 0"));
	axisLayers_.push_back(new FCLayer(cudnnHandle_, cublasHandle_, batchSize_, hiddenSize, axes, "Axes FC 1", train_, wdaxes));
	const auto outSizeBytes = (buttons_+axes_)*batchSize_*sizeof(__half);
	checkCUDA(cudaMalloc(&outData_, outSizeBytes));
	checkCUDA(cudaMemset(outData_, 0, outSizeBytes));
}
__half* CustomOutLayer::Forward(__half* data){
	auto buttonData = data;
	auto axisData = data;
	for(int i = 0; i<buttonLayers_.size(); ++i){
		//std::cout << buttonLayers_[i]->layerName_ << " ";
		buttonData = buttonLayers_[i]->Forward(buttonData);
		//PrintDataHalf(buttonData, 14, "buttonData");
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		//std::cout << axisLayers_[i]->layerName_ << " ";
		axisData = axisLayers_[i]->Forward(axisData);
		//PrintDataHalf(axisData, 8, "axisData");
	}
	checkCUDA(cudaMemcpy(outData_, buttonData, buttons_*batchSize_*sizeof(__half), cudaMemcpyDeviceToDevice));
	checkCUDA(cudaMemcpy(outData_ + buttons_*batchSize_, axisData, axes_*batchSize_*sizeof(__half), cudaMemcpyDeviceToDevice));
	//PrintDataHalf(outData_, (buttons_+axes_)*batchSize_, "outData_");
	return outData_;
}
__half* CustomOutLayer::Backward(__half* grad){ 
	auto buttonGrad = grad;
	auto axisGrad = grad + buttons_*batchSize_;
	for(int i = buttonLayers_.size(); --i >= 0; ){
		//std::cout << buttonLayers_[i]->layerName_ << " ";
		//PrintDataHalf(buttonGrad, 8, "gradient");
		buttonGrad = buttonLayers_[i]->Backward(buttonGrad);
	}
	for(int i = axisLayers_.size(); --i >= 0; ){
		//std::cout << axisLayers_[i]->layerName_ << " ";
		//PrintDataHalf(axisGrad, 8, "gradient");
		axisGrad = axisLayers_[i]->Backward(axisGrad);
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, outDesc_, axisGrad, &alpha, outDesc_, buttonGrad));
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
void CustomOutLayer::SaveParameters(std::ofstream& file, float* buffer){
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->SaveParameters(file, buffer);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->SaveParameters(file, buffer);
	}
}
void CustomOutLayer::LoadParameters(std::ifstream& file, float* buffer){ 
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->LoadParameters(file, buffer);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->LoadParameters(file, buffer);
	}
}
void CustomOutLayer::SaveOptimizerState(std::ofstream& file, float* buffer){ 
	for(int i = 0; i<buttonLayers_.size(); ++i){
		buttonLayers_[i]->SaveOptimizerState(file, buffer);
	}
	for(int i = 0; i<axisLayers_.size(); ++i){
		axisLayers_[i]->SaveOptimizerState(file, buffer);
	}
}
void CustomOutLayer::LoadOptimizerState(std::ifstream& file, float* buffer){ 
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