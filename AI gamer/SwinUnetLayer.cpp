#include "SwinUnetLayer.h"
#include "CuCommon.cuh"
#include "FCLayer.h"
#include "SwinBlockLayer.h"
#include <algorithm>
#include <stdexcept>

SwinUnetLayer::SwinUnetLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int patchRows, const int patchCols, const int embedDim, const int ffDim, const int numHeads, const int windowSize, const int shiftStride, const int depth0, const int depth1, const int depth2, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength)
	: cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), batchSize_(batchSize), patchRows_(patchRows), patchCols_(patchCols), embedDim_(embedDim), ffDim_(ffDim), numHeads_(numHeads), windowSize_(windowSize), shiftStride_(shiftStride), depth0_(depth0), depth1_(depth1), depth2_(depth2){
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
	outNCHW_ = static_cast<size_t>(batchSize_) * tokens_ * embedDim_;
	const int packedDim = embedDim_ * 4;
	CUDAMallocZero(&mergePacked_, static_cast<size_t>(batchSize_) * mergedTokens_ * packedDim * sizeof(__half));
	CUDAMallocZero(&mergePackedGrad_, static_cast<size_t>(batchSize_) * tokens_ * embedDim_ * sizeof(__half));
	CUDAMallocZero(&expandPackedGrad_, static_cast<size_t>(batchSize_) * mergedTokens_ * packedDim * sizeof(__half));
	CUDAMallocZero(&expandedTokens_, outNCHW_ * sizeof(__half));

	for(int i = 0; i < depth0_; ++i){
		layerNames_.push_back("SwinUnetEnc0_" + std::to_string(i));
		const int shiftSize = (i % 2 == 0) ? 0 : shiftStride_;
		enc0_.push_back(new SwinBlockLayer(cudnnHandle_, cublasHandle_, batchSize_, tokens_, embedDim_, ffDim_, numHeads_, patchRows_, patchCols_, windowSize_, shiftSize, layerNames_.back().c_str(), train_, weightDecay, gradAccumLength));
	}
	mergeProj_ = new FCLayer(cublasHandle_, batchSize_ * mergedTokens_, packedDim, embedDim_ * 2, "SwinUnetMergeProj", train_, weightDecay, gradAccumLength, Xavier);
	for(int i = 0; i < depth1_; ++i){
		layerNames_.push_back("SwinUnetEnc1_" + std::to_string(i));
		const int shiftSize = (i % 2 == 0) ? 0 : shiftStride_;
		enc1_.push_back(new SwinBlockLayer(cudnnHandle_, cublasHandle_, batchSize_, mergedTokens_, embedDim_ * 2, ffDim_ * 2, numHeads_, mergedRows_, mergedCols_, windowSize_, shiftSize, layerNames_.back().c_str(), train_, weightDecay, gradAccumLength));
	}
	expandProj_ = new FCLayer(cublasHandle_, batchSize_ * mergedTokens_, embedDim_ * 2, packedDim, "SwinUnetExpandProj", train_, weightDecay, gradAccumLength, Xavier);
	for(int i = 0; i < depth2_; ++i){
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
	for(const auto layer : dec0_){
		delete layer;
	}
	dec0_.clear();
	delete mergeProj_;
	delete expandProj_;
	cudaFree(mergePacked_);
	cudaFree(mergePackedGrad_);
	cudaFree(expandPackedGrad_);
	cudaFree(expandedTokens_);
}

__half* SwinUnetLayer::Forward(__half* data){
	for(const auto layer : enc0_){
		data = layer->Forward(data);
	}
	skipBuffer_ = data;
	PackTokens2x2(data, mergePacked_, batchSize_, patchRows_, patchCols_, embedDim_);
	data = mergeProj_->Forward(mergePacked_);
	for(const auto layer : enc1_){
		data = layer->Forward(data);
	}
	auto* expandPacked = expandProj_->Forward(data);
	UnpackTokens2x2(expandPacked, expandedTokens_, batchSize_, mergedRows_, mergedCols_, embedDim_);
	if(skipBuffer_){
		AddTensor(1.0f, expandedTokens_, 1.0f, skipBuffer_, static_cast<int>(outNCHW_));
	}
	data = expandedTokens_;
	for(const auto layer : dec0_){
		data = layer->Forward(data);
	}
	return data;
}

__half* SwinUnetLayer::Backward(__half* grad){
	for(int i = static_cast<int>(dec0_.size()); --i >= 0; ){
		grad = dec0_[i]->Backward(grad);
	}
	const auto* skipGrad = grad;
	PackTokens2x2(grad, expandPackedGrad_, batchSize_, patchRows_, patchCols_, embedDim_);
	grad = expandProj_->Backward(expandPackedGrad_);
	for(int i = static_cast<int>(enc1_.size()); --i >= 0; ){
		grad = enc1_[i]->Backward(grad);
	}
	grad = mergeProj_->Backward(grad);
	UnpackTokens2x2(grad, mergePackedGrad_, batchSize_, mergedRows_, mergedCols_, embedDim_);
	grad = mergePackedGrad_;
	if(skipGrad){
		AddTensor(1.0f, grad, 1.0f, skipGrad, static_cast<int>(outNCHW_));
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
	mergeProj_->UpdateParameters(learningRate);
	for(const auto layer : enc1_){
		layer->UpdateParameters(learningRate);
	}
	expandProj_->UpdateParameters(learningRate);
	for(const auto layer : dec0_){
		layer->UpdateParameters(learningRate);
	}
}

void SwinUnetLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(const auto layer : enc0_){
		layer->SaveParameters(file, buffer);
	}
	mergeProj_->SaveParameters(file, buffer);
	for(const auto layer : enc1_){
		layer->SaveParameters(file, buffer);
	}
	expandProj_->SaveParameters(file, buffer);
	for(const auto layer : dec0_){
		layer->SaveParameters(file, buffer);
	}
}

void SwinUnetLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(const auto layer : enc0_){
		layer->LoadParameters(file, buffer);
	}
	mergeProj_->LoadParameters(file, buffer);
	for(const auto layer : enc1_){
		layer->LoadParameters(file, buffer);
	}
	expandProj_->LoadParameters(file, buffer);
	for(const auto layer : dec0_){
		layer->LoadParameters(file, buffer);
	}
}

void SwinUnetLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(const auto layer : enc0_){
		layer->SaveOptimizerState(file, buffer);
	}
	mergeProj_->SaveOptimizerState(file, buffer);
	for(const auto layer : enc1_){
		layer->SaveOptimizerState(file, buffer);
	}
	expandProj_->SaveOptimizerState(file, buffer);
	for(const auto layer : dec0_){
		layer->SaveOptimizerState(file, buffer);
	}
}

void SwinUnetLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(const auto layer : enc0_){
		layer->LoadOptimizerState(file, buffer);
	}
	mergeProj_->LoadOptimizerState(file, buffer);
	for(const auto layer : enc1_){
		layer->LoadOptimizerState(file, buffer);
	}
	expandProj_->LoadOptimizerState(file, buffer);
	for(const auto layer : dec0_){
		layer->LoadOptimizerState(file, buffer);
	}
}

size_t SwinUnetLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(const auto layer : enc0_){
		maxSize = std::max(maxSize, layer->GetParameterSize());
	}
	maxSize = std::max(maxSize, mergeProj_->GetParameterSize());
	for(const auto layer : enc1_){
		maxSize = std::max(maxSize, layer->GetParameterSize());
	}
	maxSize = std::max(maxSize, expandProj_->GetParameterSize());
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
	maxSize = std::max(maxSize, mergeProj_->GetOptimizerStateSize());
	for(const auto layer : enc1_){
		maxSize = std::max(maxSize, layer->GetOptimizerStateSize());
	}
	maxSize = std::max(maxSize, expandProj_->GetOptimizerStateSize());
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
	mergeProj_->SetTrain(enable);
	for(const auto layer : enc1_){
		layer->SetTrain(enable);
	}
	expandProj_->SetTrain(enable);
	for(const auto layer : dec0_){
		layer->SetTrain(enable);
	}
}
