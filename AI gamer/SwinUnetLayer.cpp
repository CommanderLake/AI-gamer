#include "SwinUnetLayer.h"
#include "CuCommon.cuh"
#include "FCLayer.h"
#include "SwinBlockLayer.h"
#include <algorithm>
#include <stdexcept>

SwinUnetLayer::SwinUnetLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int patchRows, const int patchCols, const int embedDim, const int ffDim, const int numHeads, const int windowSize, const int shiftStride, const int depth0, const int depth1, const int depth2, const int depth3, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength)
	: cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), batchSize_(batchSize), patchRows_(patchRows), patchCols_(patchCols), embedDim_(embedDim), ffDim_(ffDim), numHeads_(numHeads), windowSize_(windowSize), shiftStride_(shiftStride), depth0_(depth0), depth1_(depth1), depth2_(depth2), depth3_(depth3){
	if(patchRows_ % 2 != 0 || patchCols_ % 2 != 0){
		throw std::invalid_argument("SwinUnetLayer requires even patch dimensions");
	}
	if(patchRows_ <= 0 || patchCols_ <= 0){
		throw std::invalid_argument("SwinUnetLayer requires positive patch dimensions");
	}
	layerName_ = layerName;
	train_ = train;
	tokens_ = patchRows_ * patchCols_;
	mergedRows_ = patchRows_ / 2;
	mergedCols_ = patchCols_ / 2;
	mergedTokens_ = mergedRows_ * mergedCols_;
	if(mergedRows_ * 2 != patchRows_ || mergedCols_ * 2 != patchCols_ || tokens_ != patchRows_ * patchCols_){
		throw std::invalid_argument("SwinUnetLayer patch merging dimensions are inconsistent");
	}
	if(mergedRows_ % 2 != 0 || mergedCols_ % 2 != 0){
		throw std::invalid_argument("SwinUnetLayer requires even merged patch dimensions for second reduction");
	}
	reducedRows_ = mergedRows_ / 2;
	reducedCols_ = mergedCols_ / 2;
	reducedTokens_ = reducedRows_ * reducedCols_;
	if(reducedRows_ * 2 != mergedRows_ || reducedCols_ * 2 != mergedCols_){
		throw std::invalid_argument("SwinUnetLayer second merge dimensions are inconsistent");
	}
	outNCHW_ = static_cast<size_t>(batchSize_) * tokens_ * embedDim_;
	const int packedDim1 = embedDim_ * 4;
	const int packedDim2 = embedDim_ * 8;
	CUDAMallocZero(&mergePacked1_, static_cast<size_t>(batchSize_) * mergedTokens_ * packedDim1 * sizeof(__half));
	CUDAMallocZero(&mergePacked1Grad_, static_cast<size_t>(batchSize_) * tokens_ * embedDim_ * sizeof(__half));
	CUDAMallocZero(&mergePacked2_, static_cast<size_t>(batchSize_) * reducedTokens_ * packedDim2 * sizeof(__half));
	CUDAMallocZero(&mergePacked2Grad_, static_cast<size_t>(batchSize_) * mergedTokens_ * (embedDim_ * 2) * sizeof(__half));
	CUDAMallocZero(&expandPacked1Grad_, static_cast<size_t>(batchSize_) * mergedTokens_ * packedDim1 * sizeof(__half));
	CUDAMallocZero(&expandPacked2Grad_, static_cast<size_t>(batchSize_) * reducedTokens_ * packedDim2 * sizeof(__half));
	CUDAMallocZero(&expandedTokens1_, outNCHW_ * sizeof(__half));
	CUDAMallocZero(&expandedTokens2_, static_cast<size_t>(batchSize_) * mergedTokens_ * (embedDim_ * 2) * sizeof(__half));

	for(int i = 0; i < depth0_; ++i){
		layerNames_.push_back("SwinUnetEnc0_" + std::to_string(i));
		const int shiftSize = (i % 2 == 0) ? 0 : shiftStride_;
		enc0_.push_back(new SwinBlockLayer(cudnnHandle_, cublasHandle_, batchSize_, tokens_, embedDim_, ffDim_, numHeads_, patchRows_, patchCols_, windowSize_, shiftSize, layerNames_.back().c_str(), train_, weightDecay, gradAccumLength));
	}
	mergeProj1_ = new FCLayer(cublasHandle_, batchSize_ * mergedTokens_, packedDim1, embedDim_ * 2, "SwinUnetMergeProj1", train_, weightDecay, gradAccumLength, Xavier);
	for(int i = 0; i < depth1_; ++i){
		layerNames_.push_back("SwinUnetEnc1_" + std::to_string(i));
		const int shiftSize = (i % 2 == 0) ? 0 : shiftStride_;
		enc1_.push_back(new SwinBlockLayer(cudnnHandle_, cublasHandle_, batchSize_, mergedTokens_, embedDim_ * 2, ffDim_ * 2, numHeads_, mergedRows_, mergedCols_, windowSize_, shiftSize, layerNames_.back().c_str(), train_, weightDecay, gradAccumLength));
	}
	mergeProj2_ = new FCLayer(cublasHandle_, batchSize_ * reducedTokens_, packedDim2, embedDim_ * 4, "SwinUnetMergeProj2", train_, weightDecay, gradAccumLength, Xavier);
	for(int i = 0; i < depth2_; ++i){
		layerNames_.push_back("SwinUnetEnc2_" + std::to_string(i));
		const int shiftSize = (i % 2 == 0) ? 0 : shiftStride_;
		enc2_.push_back(new SwinBlockLayer(cudnnHandle_, cublasHandle_, batchSize_, reducedTokens_, embedDim_ * 4, ffDim_ * 4, numHeads_, reducedRows_, reducedCols_, windowSize_, shiftSize, layerNames_.back().c_str(), train_, weightDecay, gradAccumLength));
	}
	expandProj2_ = new FCLayer(cublasHandle_, batchSize_ * reducedTokens_, embedDim_ * 4, packedDim2, "SwinUnetExpandProj2", train_, weightDecay, gradAccumLength, Xavier);
	for(int i = 0; i < depth3_; ++i){
		layerNames_.push_back("SwinUnetDec1_" + std::to_string(i));
		const int shiftSize = (i % 2 == 0) ? 0 : shiftStride_;
		dec1_.push_back(new SwinBlockLayer(cudnnHandle_, cublasHandle_, batchSize_, mergedTokens_, embedDim_ * 2, ffDim_ * 2, numHeads_, mergedRows_, mergedCols_, windowSize_, shiftSize, layerNames_.back().c_str(), train_, weightDecay, gradAccumLength));
	}
	expandProj1_ = new FCLayer(cublasHandle_, batchSize_ * mergedTokens_, embedDim_ * 2, packedDim1, "SwinUnetExpandProj1", train_, weightDecay, gradAccumLength, Xavier);
	for(int i = 0; i < depth0_; ++i){
		layerNames_.push_back("SwinUnetDec0_" + std::to_string(i));
		const int shiftSize = (i % 2 == 0) ? 0 : shiftStride_;
		dec0_.push_back(new SwinBlockLayer(cudnnHandle_, cublasHandle_, batchSize_, tokens_, embedDim_, ffDim_, numHeads_, patchRows_, patchCols_, windowSize_, shiftSize, layerNames_.back().c_str(), train_, weightDecay, gradAccumLength));
	}
}

SwinUnetLayer::~SwinUnetLayer(){
	for(const auto layer : enc0_){
		delete layer;
	}
	enc0_.clear();
	for(const auto layer : enc1_){
		delete layer;
	}
	enc1_.clear();
	for(const auto layer : enc2_){
		delete layer;
	}
	enc2_.clear();
	for(const auto layer : dec1_){
		delete layer;
	}
	dec1_.clear();
	for(const auto layer : dec0_){
		delete layer;
	}
	dec0_.clear();
	delete mergeProj1_;
	delete mergeProj2_;
	delete expandProj2_;
	delete expandProj1_;
	cudaFree(mergePacked1_);
	cudaFree(mergePacked1Grad_);
	cudaFree(mergePacked2_);
	cudaFree(mergePacked2Grad_);
	cudaFree(expandPacked1Grad_);
	cudaFree(expandPacked2Grad_);
	cudaFree(expandedTokens1_);
	cudaFree(expandedTokens2_);
}

__half* SwinUnetLayer::Forward(__half* data){
	for(const auto layer : enc0_){
		data = layer->Forward(data);
	}
	skipBuffer0_ = data;
	PackTokens2x2(data, mergePacked1_, batchSize_, patchRows_, patchCols_, embedDim_);
	data = mergeProj1_->Forward(mergePacked1_);
	for(const auto layer : enc1_){
		data = layer->Forward(data);
	}
	skipBuffer1_ = data;
	PackTokens2x2(data, mergePacked2_, batchSize_, mergedRows_, mergedCols_, embedDim_ * 2);
	data = mergeProj2_->Forward(mergePacked2_);
	for(const auto layer : enc2_){
		data = layer->Forward(data);
	}
	auto* expandPacked2 = expandProj2_->Forward(data);
	UnpackTokens2x2(expandPacked2, expandedTokens2_, batchSize_, reducedRows_, reducedCols_, embedDim_ * 2);
	if(skipBuffer1_){
		AddTensor(1.0f, expandedTokens2_, 1.0f, skipBuffer1_, static_cast<int>(static_cast<size_t>(batchSize_) * mergedTokens_ * (embedDim_ * 2)));
	}
	data = expandedTokens2_;
	for(const auto layer : dec1_){
		data = layer->Forward(data);
	}
	auto* expandPacked1 = expandProj1_->Forward(data);
	UnpackTokens2x2(expandPacked1, expandedTokens1_, batchSize_, mergedRows_, mergedCols_, embedDim_);
	if(skipBuffer0_){
		AddTensor(1.0f, expandedTokens1_, 1.0f, skipBuffer0_, static_cast<int>(outNCHW_));
	}
	data = expandedTokens1_;
	for(const auto layer : dec0_){
		data = layer->Forward(data);
	}
	return data;
}

__half* SwinUnetLayer::Backward(__half* grad){
	for(int i = static_cast<int>(dec0_.size()); --i >= 0; ){
		grad = dec0_[i]->Backward(grad);
	}
	const auto* skipGrad0 = grad;
	PackTokens2x2(grad, expandPacked1Grad_, batchSize_, patchRows_, patchCols_, embedDim_);
	grad = expandProj1_->Backward(expandPacked1Grad_);
	for(int i = static_cast<int>(dec1_.size()); --i >= 0; ){
		grad = dec1_[i]->Backward(grad);
	}
	const auto* skipGrad1 = grad;
	PackTokens2x2(grad, expandPacked2Grad_, batchSize_, mergedRows_, mergedCols_, embedDim_ * 2);
	grad = expandProj2_->Backward(expandPacked2Grad_);
	for(int i = static_cast<int>(enc2_.size()); --i >= 0; ){
		grad = enc2_[i]->Backward(grad);
	}
	grad = mergeProj2_->Backward(grad);
	UnpackTokens2x2(grad, mergePacked2Grad_, batchSize_, reducedRows_, reducedCols_, embedDim_ * 2);
	grad = mergePacked2Grad_;
	if(skipGrad1){
		AddTensor(1.0f, grad, 1.0f, skipGrad1, static_cast<int>(static_cast<size_t>(batchSize_) * mergedTokens_ * (embedDim_ * 2)));
	}
	for(int i = static_cast<int>(enc1_.size()); --i >= 0; ){
		grad = enc1_[i]->Backward(grad);
	}
	grad = mergeProj1_->Backward(grad);
	UnpackTokens2x2(grad, mergePacked1Grad_, batchSize_, mergedRows_, mergedCols_, embedDim_);
	grad = mergePacked1Grad_;
	if(skipGrad0){
		AddTensor(1.0f, grad, 1.0f, skipGrad0, static_cast<int>(outNCHW_));
	}
	for(int i = static_cast<int>(enc0_.size()); --i >= 0; ){
		grad = enc0_[i]->Backward(grad);
	}
	return grad;
}

void SwinUnetLayer::UpdateParameters(const float learningRate){
	for(const auto layer : enc0_){
		layer->UpdateParameters(learningRate);
	}
	mergeProj1_->UpdateParameters(learningRate);
	for(const auto layer : enc1_){
		layer->UpdateParameters(learningRate);
	}
	mergeProj2_->UpdateParameters(learningRate);
	for(const auto layer : enc2_){
		layer->UpdateParameters(learningRate);
	}
	expandProj2_->UpdateParameters(learningRate);
	for(const auto layer : dec1_){
		layer->UpdateParameters(learningRate);
	}
	expandProj1_->UpdateParameters(learningRate);
	for(const auto layer : dec0_){
		layer->UpdateParameters(learningRate);
	}
}

void SwinUnetLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(const auto layer : enc0_){
		layer->SaveParameters(file, buffer);
	}
	mergeProj1_->SaveParameters(file, buffer);
	for(const auto layer : enc1_){
		layer->SaveParameters(file, buffer);
	}
	mergeProj2_->SaveParameters(file, buffer);
	for(const auto layer : enc2_){
		layer->SaveParameters(file, buffer);
	}
	expandProj2_->SaveParameters(file, buffer);
	for(const auto layer : dec1_){
		layer->SaveParameters(file, buffer);
	}
	expandProj1_->SaveParameters(file, buffer);
	for(const auto layer : dec0_){
		layer->SaveParameters(file, buffer);
	}
}

void SwinUnetLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(const auto layer : enc0_){
		layer->LoadParameters(file, buffer);
	}
	mergeProj1_->LoadParameters(file, buffer);
	for(const auto layer : enc1_){
		layer->LoadParameters(file, buffer);
	}
	mergeProj2_->LoadParameters(file, buffer);
	for(const auto layer : enc2_){
		layer->LoadParameters(file, buffer);
	}
	expandProj2_->LoadParameters(file, buffer);
	for(const auto layer : dec1_){
		layer->LoadParameters(file, buffer);
	}
	expandProj1_->LoadParameters(file, buffer);
	for(const auto layer : dec0_){
		layer->LoadParameters(file, buffer);
	}
}

void SwinUnetLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(const auto layer : enc0_){
		layer->SaveOptimizerState(file, buffer);
	}
	mergeProj1_->SaveOptimizerState(file, buffer);
	for(const auto layer : enc1_){
		layer->SaveOptimizerState(file, buffer);
	}
	mergeProj2_->SaveOptimizerState(file, buffer);
	for(const auto layer : enc2_){
		layer->SaveOptimizerState(file, buffer);
	}
	expandProj2_->SaveOptimizerState(file, buffer);
	for(const auto layer : dec1_){
		layer->SaveOptimizerState(file, buffer);
	}
	expandProj1_->SaveOptimizerState(file, buffer);
	for(const auto layer : dec0_){
		layer->SaveOptimizerState(file, buffer);
	}
}

void SwinUnetLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(const auto layer : enc0_){
		layer->LoadOptimizerState(file, buffer);
	}
	mergeProj1_->LoadOptimizerState(file, buffer);
	for(const auto layer : enc1_){
		layer->LoadOptimizerState(file, buffer);
	}
	mergeProj2_->LoadOptimizerState(file, buffer);
	for(const auto layer : enc2_){
		layer->LoadOptimizerState(file, buffer);
	}
	expandProj2_->LoadOptimizerState(file, buffer);
	for(const auto layer : dec1_){
		layer->LoadOptimizerState(file, buffer);
	}
	expandProj1_->LoadOptimizerState(file, buffer);
	for(const auto layer : dec0_){
		layer->LoadOptimizerState(file, buffer);
	}
}

size_t SwinUnetLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(const auto layer : enc0_){
		maxSize = std::max(maxSize, layer->GetParameterSize());
	}
	maxSize = std::max(maxSize, mergeProj1_->GetParameterSize());
	for(const auto layer : enc1_){
		maxSize = std::max(maxSize, layer->GetParameterSize());
	}
	maxSize = std::max(maxSize, mergeProj2_->GetParameterSize());
	for(const auto layer : enc2_){
		maxSize = std::max(maxSize, layer->GetParameterSize());
	}
	maxSize = std::max(maxSize, expandProj2_->GetParameterSize());
	for(const auto layer : dec1_){
		maxSize = std::max(maxSize, layer->GetParameterSize());
	}
	maxSize = std::max(maxSize, expandProj1_->GetParameterSize());
	for(const auto layer : dec0_){
		maxSize = std::max(maxSize, layer->GetParameterSize());
	}
	return maxSize;
}

size_t SwinUnetLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(const auto layer : enc0_){
		maxSize = std::max(maxSize, layer->GetOptimizerStateSize());
	}
	maxSize = std::max(maxSize, mergeProj1_->GetOptimizerStateSize());
	for(const auto layer : enc1_){
		maxSize = std::max(maxSize, layer->GetOptimizerStateSize());
	}
	maxSize = std::max(maxSize, mergeProj2_->GetOptimizerStateSize());
	for(const auto layer : enc2_){
		maxSize = std::max(maxSize, layer->GetOptimizerStateSize());
	}
	maxSize = std::max(maxSize, expandProj2_->GetOptimizerStateSize());
	for(const auto layer : dec1_){
		maxSize = std::max(maxSize, layer->GetOptimizerStateSize());
	}
	maxSize = std::max(maxSize, expandProj1_->GetOptimizerStateSize());
	for(const auto layer : dec0_){
		maxSize = std::max(maxSize, layer->GetOptimizerStateSize());
	}
	return maxSize;
}

void SwinUnetLayer::SetTrain(const bool enable){
	train_ = enable;
	for(const auto layer : enc0_){
		layer->SetTrain(enable);
	}
	mergeProj1_->SetTrain(enable);
	for(const auto layer : enc1_){
		layer->SetTrain(enable);
	}
	mergeProj2_->SetTrain(enable);
	for(const auto layer : enc2_){
		layer->SetTrain(enable);
	}
	expandProj2_->SetTrain(enable);
	for(const auto layer : dec1_){
		layer->SetTrain(enable);
	}
	expandProj1_->SetTrain(enable);
	for(const auto layer : dec0_){
		layer->SetTrain(enable);
	}
}
