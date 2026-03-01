#include "PatchExpandingLayer.h"
#include "CuCommon.cuh"
#include "FCLayer.h"
#include "LayerNorm.h"
#include <algorithm>
PatchExpandingLayer::PatchExpandingLayer(const cudnnHandle_t cudnnHandle, const int batchSize, const int tokens, const int embedDim, const int patchRows, const int patchCols, const std::string layerName, const bool train,
	const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod) : cudnnHandle_(cudnnHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), patchRows_(patchRows), patchCols_(patchCols){
	layerName_ = layerName;
	train_ = train;
	if(tokens_ != patchRows_*patchCols_){ throw std::invalid_argument("PatchExpandingLayer tokens must match patch grid"); }
	if(embedDim_ % 2 != 0){ throw std::invalid_argument("PatchExpandingLayer requires even embedDim"); }
	outPatchRows_ = patchRows_ * 2;
	outPatchCols_ = patchCols_ * 2;
	outTokens_ = outPatchRows_ * outPatchCols_;
	outEmbedDim_ = embedDim_ / 2;
	outNCHW_ = static_cast<size_t>(batchSize_) * outTokens_ * outEmbedDim_;
	norm_ = new LayerNorm(batchSize_*tokens_, embedDim_, 1, 1, "PatchExpandNorm", train);
	expansion_ = new FCLayer(batchSize_*tokens_, embedDim_, outEmbedDim_*4, "PatchExpandLinear", train, weightDecay, gradAccumLength, weightInitMethod);
	const size_t expandedElems = static_cast<size_t>(batchSize_) * tokens_ * outEmbedDim_ * 4;
	CUDAMallocZero(&expandedData_, expandedElems*sizeof(__half));
	CUDAMallocZero(&expandedTokens_, static_cast<size_t>(batchSize_) * outTokens_ * outEmbedDim_ * sizeof(__half));
	CUDAMallocZero(&mergedGrad_, expandedElems*sizeof(__half));
}

PatchExpandingLayer::~PatchExpandingLayer(){
	delete norm_;
	delete expansion_;
	cudaFree(expandedData_);
	cudaFree(expandedTokens_);
	cudaFree(mergedGrad_);
}

__half* PatchExpandingLayer::Forward(__half* data){
	data = norm_->Forward(data);
	data = expansion_->Forward(data);
	PatchUnmerge(data, expandedTokens_, batchSize_, outTokens_, outEmbedDim_, outPatchRows_, outPatchCols_);
	return expandedTokens_;
}

__half* PatchExpandingLayer::Backward(__half* grad){
	PatchMerge(grad, mergedGrad_, batchSize_, outTokens_, outEmbedDim_, outPatchRows_, outPatchCols_);
	grad = expansion_->Backward(mergedGrad_);
	return norm_->Backward(grad);
}

void PatchExpandingLayer::UpdateParameters(const float lr){
	norm_->UpdateParameters(lr);
	expansion_->UpdateParameters(lr);
}

void PatchExpandingLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	norm_->SaveParameters(file, buffer);
	expansion_->SaveParameters(file, buffer);
}

void PatchExpandingLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	norm_->LoadParameters(file, buffer);
	expansion_->LoadParameters(file, buffer);
}

void PatchExpandingLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	norm_->SaveOptimizerState(file, buffer);
	expansion_->SaveOptimizerState(file, buffer);
}

void PatchExpandingLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	norm_->LoadOptimizerState(file, buffer);
	expansion_->LoadOptimizerState(file, buffer);
}

size_t PatchExpandingLayer::GetParameterSize(){ return std::max(norm_->GetParameterSize(), expansion_->GetParameterSize()); }

size_t PatchExpandingLayer::GetOptimizerStateSize(){ return std::max(norm_->GetOptimizerStateSize(), expansion_->GetOptimizerStateSize()); }

void PatchExpandingLayer::SetTrain(const bool enable){
	train_ = enable;
	norm_->SetTrain(enable);
	expansion_->SetTrain(enable);
}