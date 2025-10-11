#include "GlobalPoolLayer.h"
#include "CuCommon.cuh"
GlobalPoolLayer::GlobalPoolLayer(int batchSize, int tokens, int embedDim, const char* layerName, bool train) : ogbs_(batchSize), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim){
	layerName_ = layerName;
	train_ = train;
	invSqrtDim_ = 1.0f / std::sqrt(static_cast<float>(embedDim_));
	outNCHW_ = batchSize_ * embedDim_;
	weightCount_ = embedDim_;
	CUDAMallocZero(&outData_, outNCHW_ * sizeof(__half));
	CUDAMallocZero(&query_, weightCount_ * sizeof(__half));
	CUDAMallocZero(&attnWeights_, batchSize_ * tokens_ * sizeof(float));
	CUDAMallocZero(&scratchBuffer_, batchSize_ * tokens_ * sizeof(float));
	CUDAMallocZero(&batchSums_, batchSize_ * sizeof(float));
	if(train_){
		CUDAMallocZero(&outGrad_, batchSize_ * tokens_ * embedDim_ * sizeof(__half));
		CUDAMallocZero(&gradQuery_, weightCount_ * sizeof(__half));
		CUDAMallocZero(&mQuery_, weightCount_ * sizeof(__half));
		CUDAMallocZero(&vQuery_, weightCount_ * sizeof(__half));
	}
	weights_ = query_;
}
GlobalPoolLayer::~GlobalPoolLayer(){
	cudaFree(outData_);
	cudaFree(query_);
	cudaFree(attnWeights_);
	cudaFree(scratchBuffer_);
	cudaFree(batchSums_);
	if(outGrad_){ cudaFree(outGrad_); }
	if(gradQuery_){ cudaFree(gradQuery_); }
	if(mQuery_){ cudaFree(mQuery_); }
	if(vQuery_){ cudaFree(vQuery_); }
}
__half* GlobalPoolLayer::Forward(__half* data){
	inData_ = data;
	AttentionPoolForward(data, query_, outData_, attnWeights_, scratchBuffer_, batchSize_, tokens_, embedDim_, invSqrtDim_);
	return outData_;
}
__half* GlobalPoolLayer::Backward(__half* grad){
	if(!train_ || !outGrad_){ return grad; }
	AttentionPoolBackward(grad, inData_, query_, attnWeights_, scratchBuffer_, batchSums_, outGrad_, gradQuery_, batchSize_, tokens_, embedDim_, invSqrtDim_);
	return outGrad_;
}
void GlobalPoolLayer::UpdateParameters(float learningRate){
	if(!train_ || !gradQuery_){ return; }
	AdamWHalf(query_, gradQuery_, mQuery_, vQuery_, learningRate, t_, weightDecay_, embedDim_);
	++t_;
}
void GlobalPoolLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, query_, weightCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_ * sizeof(__half));
}
void GlobalPoolLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(query_, buffer, weightCount_ * sizeof(__half), cudaMemcpyHostToDevice);
}
void GlobalPoolLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!train_ || !mQuery_ || !vQuery_){ return; }
	cudaMemcpy(buffer, mQuery_, weightCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(buffer, vQuery_, weightCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_ * sizeof(__half));
}
void GlobalPoolLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!train_ || !mQuery_ || !vQuery_){ return; }
	file.read(reinterpret_cast<char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(mQuery_, buffer, weightCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(vQuery_, buffer, weightCount_ * sizeof(__half), cudaMemcpyHostToDevice);
}
size_t GlobalPoolLayer::GetParameterSize(){ return weightCount_ * sizeof(__half); }
size_t GlobalPoolLayer::GetOptimizerStateSize(){
	if(!train_ || !mQuery_ || !vQuery_){ return 0; }
	return 2 * weightCount_ * sizeof(__half);
}
void GlobalPoolLayer::SetTrain(bool enable){
	if(enable && !train_){
		if(!outGrad_){ CUDAMallocZero(&outGrad_, ogbs_ * tokens_ * embedDim_ * sizeof(__half)); }
		if(!gradQuery_){ CUDAMallocZero(&gradQuery_, weightCount_ * sizeof(__half)); }
		if(!mQuery_){ CUDAMallocZero(&mQuery_, weightCount_ * sizeof(__half)); }
		if(!vQuery_){ CUDAMallocZero(&vQuery_, weightCount_ * sizeof(__half)); }
	}
	train_ = enable;
	batchSize_ = enable ? ogbs_ : 1;
	outNCHW_ = batchSize_ * embedDim_;
}