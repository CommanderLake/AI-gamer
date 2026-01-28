#include "DropPath.h"
#include "CuCommon.cuh"
#include <stdexcept>

DropPath::DropPath(const float dropRate, const int batchSize, const int elementsPerBatch, const char* layerName, const bool train)
	: dropRate_(dropRate),
	  keepProb_(1.0f - dropRate),
	  batchSize_(batchSize),
	  elementsPerBatch_(elementsPerBatch){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = static_cast<size_t>(batchSize_)*elementsPerBatch_;
	if(dropRate_ > 0.0f){
		CUDAMallocZero(&mask_, static_cast<size_t>(batchSize_)*sizeof(float));
	}
}

DropPath::~DropPath(){
	cudaFree(mask_);
}

__half* DropPath::Forward(__half* data){
	if(!train_ || dropRate_ <= 0.0f){ return data; }
	const auto status = curandGenerateUniform(generator_, mask_, batchSize_);
	if(status != CURAND_STATUS_SUCCESS){
		throw std::runtime_error("curandGenerateUniform failed in DropPath::Forward");
	}
	DropPathBuildMask(mask_, batchSize_, keepProb_);
	DropPathApply(data, mask_, batchSize_, elementsPerBatch_);
	return data;
}

__half* DropPath::Backward(__half* grad){
	if(!train_ || dropRate_ <= 0.0f){ return grad; }
	DropPathApply(grad, mask_, batchSize_, elementsPerBatch_);
	return grad;
}

void DropPath::SetTrain(const bool enable){
	train_ = enable;
}
