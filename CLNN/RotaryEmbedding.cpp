#include "RotaryEmbedding.h"
#include "CuCommon.h"
#include <stdexcept>
RotaryEmbedding::RotaryEmbedding(const int batchSize, const int tokens, const int embedDim, const int numHeads, int rotaryDim, std::string layerName, const float theta, const bool interleaved)
	: batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), numHeads_(numHeads), theta_(theta), interleaved_(interleaved){
	if(batchSize_ <= 0 || tokens_ <= 0 || embedDim_ <= 0 || numHeads_ <= 0){ throw std::invalid_argument("RotaryEmbedding dimensions must be positive"); }
	if(embedDim_%numHeads_ != 0){ throw std::invalid_argument("RotaryEmbedding embedDim must be divisible by numHeads"); }
	if(theta_ <= 1.0f){ throw std::invalid_argument("RotaryEmbedding theta must be greater than 1"); }
	headDim_ = embedDim_/numHeads_;
	if(rotaryDim <= 0){ rotaryDim = headDim_; }
	if(rotaryDim%2 != 0 || rotaryDim > headDim_){ throw std::invalid_argument("RotaryEmbedding rotaryDim must be even and no larger than headDim"); }
	rotaryDim_ = rotaryDim;
	layerName_ = layerName;
	train_ = false;
	outNCHW_ = static_cast<size_t>(batchSize_)*tokens_*embedDim_;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&outGrad_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&ownedPositionOffsets_, static_cast<size_t>(batchSize_)*sizeof(int));
}
RotaryEmbedding::~RotaryEmbedding(){
	cudaFree(outData_);
	cudaFree(outGrad_);
	cudaFree(ownedPositionOffsets_);
}
void RotaryEmbedding::SetPositionOffset(const int offset){
	basePosition_ = offset;
	positionOffsets_ = nullptr;
}
void RotaryEmbedding::SetPositionOffsetsDevice(const int* offsets){
	if(!offsets){ throw std::invalid_argument("RotaryEmbedding received null device offsets"); }
	positionOffsets_ = offsets;
}
void RotaryEmbedding::SetPositionOffsetsHost(const int* offsets){
	if(!offsets){ throw std::invalid_argument("RotaryEmbedding received null host offsets"); }
	checkCUDA(cudaMemcpy(ownedPositionOffsets_, offsets, static_cast<size_t>(batchSize_)*sizeof(int), cudaMemcpyHostToDevice));
	positionOffsets_ = ownedPositionOffsets_;
}
__half* RotaryEmbedding::Forward(__half* data){
	if(data == nullptr){ throw std::invalid_argument("RotaryEmbedding::Forward received null input"); }
	RotaryEmbeddingForward(outData_, data, positionOffsets_, batchSize_, tokens_, embedDim_, numHeads_, rotaryDim_, basePosition_, theta_, interleaved_, false);
	return outData_;
}
__half* RotaryEmbedding::Backward(__half* grad){
	if(grad == nullptr){ throw std::invalid_argument("RotaryEmbedding::Backward received null gradient"); }
	RotaryEmbeddingForward(outGrad_, grad, positionOffsets_, batchSize_, tokens_, embedDim_, numHeads_, rotaryDim_, basePosition_, theta_, interleaved_, true);
	return outGrad_;
}
size_t RotaryEmbedding::GetParameterSize(){
	return 0;
}
size_t RotaryEmbedding::GetOptimizerStateSize(){
	return 0;
}
