#include "PatchMergingLayer.h"
#include "CuCommon.cuh"
#include "FCLayer.h"
#include "LayerNorm.h"
#include <algorithm>
PatchMergingLayer::PatchMergingLayer(const cudnnHandle_t cudnnHandle, const int batchSize, const int tokens, const int embedDim, const int patchRows, const int patchCols, const std::string layerName, const bool train, const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod) : cudnnHandle_(cudnnHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), patchRows_(patchRows), patchCols_(patchCols){
	layerName_ = layerName;
	train_ = train;
	if(tokens_ != patchRows_*patchCols_){ throw std::invalid_argument("PatchMergingLayer tokens must match patch grid"); }
	if(patchRows_ % 2 != 0 || patchCols_ % 2 != 0){ throw std::invalid_argument("PatchMergingLayer requires even patch rows/cols"); }
	outTokens_ = patchRows_/2*(patchCols_/2);
	outEmbedDim_ = embedDim_*2;
	outNCHW_ = batchSize_*outTokens_*outEmbedDim_;
	norm_ = new LayerNorm(batchSize_*tokens_, embedDim_, 1, 1, "PatchMergeNorm", train);
	reduction_ = new FCLayer(batchSize_*outTokens_, embedDim_*4, outEmbedDim_, "PatchMergeLinear", train, weightDecay, gradAccumLength, weightInitMethod);
	const size_t mergeElems = static_cast<size_t>(batchSize_)*outTokens_*embedDim_*4;
	CUDAMallocZero(&mergedData_, mergeElems*sizeof(__half));
	CUDAMallocZero(&mergedGrad_, mergeElems*sizeof(__half));
	CUDAMallocZero(&tokenGrad_, static_cast<size_t>(batchSize_)*tokens_*embedDim_*sizeof(__half));
}
PatchMergingLayer::~PatchMergingLayer(){
	delete norm_;
	delete reduction_;
	cudaFree(mergedData_);
	cudaFree(mergedGrad_);
	cudaFree(tokenGrad_);
}
__half* PatchMergingLayer::Forward(__half* data){
	data = norm_->Forward(data);
	PatchMerge(data, mergedData_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_);
	return reduction_->Forward(mergedData_);
}
__half* PatchMergingLayer::Backward(__half* grad){
	grad = reduction_->Backward(grad);
	PatchUnmerge(grad, tokenGrad_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_);
	return norm_->Backward(tokenGrad_);
}
void PatchMergingLayer::UpdateParameters(const float lr){
	norm_->UpdateParameters(lr);
	reduction_->UpdateParameters(lr);
}
void PatchMergingLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	norm_->SaveParameters(file, buffer);
	reduction_->SaveParameters(file, buffer);
}
void PatchMergingLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	norm_->LoadParameters(file, buffer);
	reduction_->LoadParameters(file, buffer);
}
void PatchMergingLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	norm_->SaveOptimizerState(file, buffer);
	reduction_->SaveOptimizerState(file, buffer);
}
void PatchMergingLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	norm_->LoadOptimizerState(file, buffer);
	reduction_->LoadOptimizerState(file, buffer);
}
size_t PatchMergingLayer::GetParameterSize(){ return std::max(norm_->GetParameterSize(), reduction_->GetParameterSize()); }
size_t PatchMergingLayer::GetOptimizerStateSize(){ return std::max(norm_->GetOptimizerStateSize(), reduction_->GetOptimizerStateSize()); }
void PatchMergingLayer::SetTrain(const bool enable){
	train_ = enable;
	norm_->SetTrain(enable);
	reduction_->SetTrain(enable);
}

void PatchMergingLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	norm_->CollectAdamWTasks(halfTasks, floatTasks);
	reduction_->CollectAdamWTasks(halfTasks, floatTasks);
}
