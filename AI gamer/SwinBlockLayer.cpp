#include "SwinBlockLayer.h"
#include "CuCommon.cuh"
#include "Dropout.h"
#include "FCLayer.h"
#include "GELULayer.h"
#include "LayerNorm.h"
#include "WmmaAttentionLayer.h"
#include <algorithm>

SwinBlockLayer::SwinBlockLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int tokens, const int embedDim, const int ffDim, const int numHeads, const int patchRows, const int patchCols, const int windowHeight, const int windowWidth, const int shiftHeight, const int shiftWidth, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod)
	: cudnnHandle_(cudnnHandle),
	cublasHandle_(cublasHandle),
	batchSize_(batchSize),
	tokens_(tokens),
	embedDim_(embedDim),
	ffDim_(ffDim),
	numHeads_(numHeads),
	patchRows_(patchRows),
	patchCols_(patchCols),
	windowHeight_(windowHeight),
	windowWidth_(windowWidth),
	shiftHeight_(shiftHeight),
	shiftWidth_(shiftWidth){
	layerName_ = layerName;
	train_ = train;
	if(tokens_ != patchRows_*patchCols_){
		throw std::invalid_argument("SwinBlockLayer tokens must match patch grid");
	}
	if(patchRows_ % windowHeight_ != 0 || patchCols_ % windowWidth_ != 0){
		throw std::invalid_argument("SwinBlockLayer window size must evenly divide patch rows/cols");
	}
	windowTokens_ = windowHeight_*windowWidth_;
	windowCount_ = (patchRows_ / windowHeight_)*(patchCols_ / windowWidth_);
	windowBatch_ = batchSize_*windowCount_;
	outNCHW_ = batchSize_*tokens_*embedDim_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_*tokens_, embedDim_, 1, 1));
	norm1_ = new LayerNorm(batchSize_*tokens_, embedDim_, 1, 1, "SwinNorm1", train);
	attention_ = new WmmaAttentionLayer(cudnnHandle_, cublasHandle_, windowBatch_, windowTokens_, embedDim_, numHeads_, "SwinAttention", train, weightDecay, gradAccumLength, weightInitMethod);
	attnDrop_ = new Dropout(cudnnHandle_, 0.1f, batchSize_*tokens_, embedDim_, 1, 1, "SwinAttnDropout", train);
	norm2_ = new LayerNorm(batchSize_*tokens_, embedDim_, 1, 1, "SwinNorm2", train);
	fc1_ = new FCLayer(cublasHandle_, batchSize_*tokens_, embedDim_, ffDim_, "SwinFC1", train, weightDecay, gradAccumLength, weightInitMethod);
	gelu_ = new GELULayer(batchSize_*tokens_, ffDim_, 1, 1, "SwinGELU");
	fc2_ = new FCLayer(cublasHandle_, batchSize_*tokens_, ffDim_, embedDim_, "SwinFC2", train, weightDecay, gradAccumLength, weightInitMethod);
	ffDrop_ = new Dropout(cudnnHandle_, 0.1f, batchSize_*tokens_, embedDim_, 1, 1, "SwinFfDropout", train);
	const size_t windowElems = static_cast<size_t>(windowBatch_)*windowTokens_*embedDim_;
	CUDAMallocZero(&windowedInput_, windowElems*sizeof(__half));
	CUDAMallocZero(&windowedGrad_, windowElems*sizeof(__half));
	CUDAMallocZero(&tokenBuffer_, static_cast<size_t>(batchSize_)*tokens_*embedDim_*sizeof(__half));
}

SwinBlockLayer::~SwinBlockLayer(){
	delete norm1_;
	delete attention_;
	delete attnDrop_;
	delete norm2_;
	delete fc1_;
	delete gelu_;
	delete fc2_;
	delete ffDrop_;
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	cudaFree(windowedInput_);
	cudaFree(windowedGrad_);
	cudaFree(tokenBuffer_);
}

__half* SwinBlockLayer::Forward(__half* data){
	const auto* residual1 = data;
	data = norm1_->Forward(data);
	TokensToWindows(data, windowedInput_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	data = attention_->Forward(windowedInput_);
	WindowsToTokens(data, tokenBuffer_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	data = attnDrop_->Forward(tokenBuffer_);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixFwd_, outDesc_, residual1, &mixFwd_, outDesc_, data));
	const auto* residual2 = data;
	data = norm2_->Forward(data);
	data = fc1_->Forward(data);
	data = gelu_->Forward(data);
	data = fc2_->Forward(data);
	data = ffDrop_->Forward(data);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixFwd_, outDesc_, residual2, &mixFwd_, outDesc_, data));
	return data;
}

__half* SwinBlockLayer::Backward(__half* grad){
	const auto* residual2 = grad;
	grad = ffDrop_->Backward(grad);
	grad = fc2_->Backward(grad);
	grad = gelu_->Backward(grad);
	grad = fc1_->Backward(grad);
	grad = norm2_->Backward(grad);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixBwd_, outDesc_, residual2, &mixBwd_, outDesc_, grad));
	const auto* residual1 = grad;
	grad = attnDrop_->Backward(grad);
	TokensToWindows(grad, windowedGrad_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	grad = attention_->Backward(windowedGrad_);
	WindowsToTokens(grad, tokenBuffer_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	grad = norm1_->Backward(tokenBuffer_);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixBwd_, outDesc_, residual1, &mixBwd_, outDesc_, grad));
	return grad;
}

void SwinBlockLayer::UpdateParameters(const float lr){
	norm1_->UpdateParameters(lr);
	attention_->UpdateParameters(lr);
	attnDrop_->UpdateParameters(lr);
	norm2_->UpdateParameters(lr);
	fc1_->UpdateParameters(lr);
	gelu_->UpdateParameters(lr);
	fc2_->UpdateParameters(lr);
	ffDrop_->UpdateParameters(lr);
}

void SwinBlockLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	norm1_->SaveParameters(file, buffer);
	attention_->SaveParameters(file, buffer);
	attnDrop_->SaveParameters(file, buffer);
	norm2_->SaveParameters(file, buffer);
	fc1_->SaveParameters(file, buffer);
	gelu_->SaveParameters(file, buffer);
	fc2_->SaveParameters(file, buffer);
	ffDrop_->SaveParameters(file, buffer);
}

void SwinBlockLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	norm1_->LoadParameters(file, buffer);
	attention_->LoadParameters(file, buffer);
	attnDrop_->LoadParameters(file, buffer);
	norm2_->LoadParameters(file, buffer);
	fc1_->LoadParameters(file, buffer);
	gelu_->LoadParameters(file, buffer);
	fc2_->LoadParameters(file, buffer);
	ffDrop_->LoadParameters(file, buffer);
}

void SwinBlockLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	norm1_->SaveOptimizerState(file, buffer);
	attention_->SaveOptimizerState(file, buffer);
	attnDrop_->SaveOptimizerState(file, buffer);
	norm2_->SaveOptimizerState(file, buffer);
	fc1_->SaveOptimizerState(file, buffer);
	gelu_->SaveOptimizerState(file, buffer);
	fc2_->SaveOptimizerState(file, buffer);
	ffDrop_->SaveOptimizerState(file, buffer);
}

void SwinBlockLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	norm1_->LoadOptimizerState(file, buffer);
	attention_->LoadOptimizerState(file, buffer);
	attnDrop_->LoadOptimizerState(file, buffer);
	norm2_->LoadOptimizerState(file, buffer);
	fc1_->LoadOptimizerState(file, buffer);
	gelu_->LoadOptimizerState(file, buffer);
	fc2_->LoadOptimizerState(file, buffer);
	ffDrop_->LoadOptimizerState(file, buffer);
}

size_t SwinBlockLayer::GetParameterSize(){
	size_t maxSize = 0;
	maxSize = std::max(maxSize, norm1_->GetParameterSize());
	maxSize = std::max(maxSize, attention_->GetParameterSize());
	maxSize = std::max(maxSize, attnDrop_->GetParameterSize());
	maxSize = std::max(maxSize, norm2_->GetParameterSize());
	maxSize = std::max(maxSize, fc1_->GetParameterSize());
	maxSize = std::max(maxSize, gelu_->GetParameterSize());
	maxSize = std::max(maxSize, fc2_->GetParameterSize());
	maxSize = std::max(maxSize, ffDrop_->GetParameterSize());
	return maxSize;
}

size_t SwinBlockLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	maxSize = std::max(maxSize, norm1_->GetOptimizerStateSize());
	maxSize = std::max(maxSize, attention_->GetOptimizerStateSize());
	maxSize = std::max(maxSize, attnDrop_->GetOptimizerStateSize());
	maxSize = std::max(maxSize, norm2_->GetOptimizerStateSize());
	maxSize = std::max(maxSize, fc1_->GetOptimizerStateSize());
	maxSize = std::max(maxSize, gelu_->GetOptimizerStateSize());
	maxSize = std::max(maxSize, fc2_->GetOptimizerStateSize());
	maxSize = std::max(maxSize, ffDrop_->GetOptimizerStateSize());
	return maxSize;
}

void SwinBlockLayer::SetTrain(const bool enable){
	train_ = enable;
	norm1_->SetTrain(enable);
	attention_->SetTrain(enable);
	attnDrop_->SetTrain(enable);
	norm2_->SetTrain(enable);
	fc1_->SetTrain(enable);
	gelu_->SetTrain(enable);
	fc2_->SetTrain(enable);
	ffDrop_->SetTrain(enable);
}
