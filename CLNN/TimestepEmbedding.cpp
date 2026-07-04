#include "TimestepEmbedding.h"
#include "CuCommon.h"
#include <stdexcept>
TimestepEmbedding::TimestepEmbedding(const int batchSize, const int embeddingDim, std::string layerName, const float maxPeriod)
	: batchSize_(batchSize), embeddingDim_(embeddingDim), maxPeriod_(maxPeriod){
	if(batchSize_ <= 0 || embeddingDim_ <= 0){ throw std::invalid_argument("TimestepEmbedding dimensions must be positive"); }
	if(maxPeriod_ <= 1.0f){ throw std::invalid_argument("TimestepEmbedding maxPeriod must be greater than 1"); }
	layerName_ = layerName;
	train_ = false;
	outNCHW_ = static_cast<size_t>(batchSize_)*embeddingDim_;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&gradHalfTimesteps_, static_cast<size_t>(batchSize_)*sizeof(__half));
	CUDAMallocZero(&ownedTimesteps_, static_cast<size_t>(batchSize_)*sizeof(float));
	CUDAMallocZero(&gradTimesteps_, static_cast<size_t>(batchSize_)*sizeof(float));
}
TimestepEmbedding::~TimestepEmbedding(){
	cudaFree(outData_);
	cudaFree(gradHalfTimesteps_);
	cudaFree(ownedTimesteps_);
	cudaFree(gradTimesteps_);
}
void TimestepEmbedding::SetTimestepsDevice(const float* timesteps){
	if(!timesteps){ throw std::invalid_argument("TimestepEmbedding received null device timesteps"); }
	timesteps_ = timesteps;
}
void TimestepEmbedding::SetTimestepsHost(const float* timesteps){
	if(!timesteps){ throw std::invalid_argument("TimestepEmbedding received null host timesteps"); }
	checkCUDA(cudaMemcpy(ownedTimesteps_, timesteps, static_cast<size_t>(batchSize_)*sizeof(float), cudaMemcpyHostToDevice));
	timesteps_ = ownedTimesteps_;
}
void TimestepEmbedding::ClearTimesteps(){
	timesteps_ = nullptr;
}
__half* TimestepEmbedding::Forward(__half* data){
	if(timesteps_){
		lastForwardUsedHalf_ = false;
		lastFloatTimesteps_ = timesteps_;
		lastHalfTimesteps_ = nullptr;
		TimestepEmbeddingForward(outData_, timesteps_, batchSize_, embeddingDim_, maxPeriod_);
	} else if(data){
		lastForwardUsedHalf_ = true;
		lastFloatTimesteps_ = nullptr;
		lastHalfTimesteps_ = data;
		TimestepEmbeddingForwardHalf(outData_, data, batchSize_, embeddingDim_, maxPeriod_);
	} else{
		throw std::runtime_error("TimestepEmbedding::Forward requires timesteps");
	}
	return outData_;
}
__half* TimestepEmbedding::Backward(__half* grad){
	if(grad == nullptr){ return nullptr; }
	if(lastForwardUsedHalf_){
		if(lastHalfTimesteps_ == nullptr){ throw std::runtime_error("TimestepEmbedding::Backward requires a previous half-timestep Forward call"); }
		TimestepEmbeddingBackwardHalf(gradHalfTimesteps_, grad, lastHalfTimesteps_, batchSize_, embeddingDim_, maxPeriod_);
		return gradHalfTimesteps_;
	}
	if(lastFloatTimesteps_){ TimestepEmbeddingBackward(gradTimesteps_, grad, lastFloatTimesteps_, batchSize_, embeddingDim_, maxPeriod_); }
	return grad;
}
float* TimestepEmbedding::GetTimestepGrad(){
	return gradTimesteps_;
}
size_t TimestepEmbedding::GetParameterSize(){
	return 0;
}
size_t TimestepEmbedding::GetOptimizerStateSize(){
	return 0;
}
