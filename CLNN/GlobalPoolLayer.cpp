#include "GlobalPoolLayer.h"
#include "CuCommon.cuh"
GlobalPoolLayer::GlobalPoolLayer(int batchSize, int nTokens, int embedSize, int numQueries, std::string layerName, bool train) : batchSize_(batchSize), nTokens_(nTokens), embedSize_(embedSize), numQueries_(numQueries){
	layerName_ = layerName;
	train_ = train;
	invSqrtDim_ = 1.0f/std::sqrt(static_cast<float>(embedSize_));
	outNCHW_ = static_cast<size_t>(batchSize_)*numQueries_*embedSize_;
	weightCount_ = static_cast<size_t>(numQueries_)*embedSize_;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	CUDAMallocZero(&attnWeights_, static_cast<size_t>(batchSize_)*numQueries_*nTokens_*sizeof(float));
	CUDAMallocZero(&scratchBuffer_, static_cast<size_t>(batchSize_)*numQueries_*nTokens_*sizeof(float));
	if(train_){
		WeightInit(weights_, static_cast<int>(weightCount_), embedSize_, embedSize_, Xavier);
		CUDAMallocZero(&outGrad_, static_cast<size_t>(batchSize_)*nTokens_*embedSize_*sizeof(__half));
		CUDAMallocZero(&gradQuery_, weightCount_*sizeof(__half));
		CUDAMallocZero(&mQuery_, weightCount_*sizeof(__half));
		CUDAMallocZero(&vQuery_, weightCount_*sizeof(__half));
		CUDAMallocZero(&batchSums_, static_cast<size_t>(batchSize_)*numQueries_*sizeof(float));
	}
}
GlobalPoolLayer::~GlobalPoolLayer(){
	cudaFree(outData_);
	cudaFree(weights_);
	cudaFree(attnWeights_);
	cudaFree(scratchBuffer_);
	if(train_){
		cudaFree(outGrad_);
		cudaFree(gradQuery_);
		cudaFree(mQuery_);
		cudaFree(vQuery_);
		cudaFree(batchSums_);
	}
}
__half* GlobalPoolLayer::Forward(__half* data){
	inData_ = data;
	AttentionPoolForward(data, weights_, outData_, attnWeights_, scratchBuffer_, batchSize_, nTokens_, embedSize_, numQueries_, invSqrtDim_);
	return outData_;
}
__half* GlobalPoolLayer::Backward(__half* grad){
	AttentionPoolBackward(grad, inData_, weights_, attnWeights_, scratchBuffer_, batchSums_, outGrad_, gradQuery_, batchSize_, nTokens_, embedSize_, numQueries_, invSqrtDim_);
	return outGrad_;
}
void GlobalPoolLayer::UpdateParameters(float learningRate){
	AdamWHalf(weights_, gradQuery_, mQuery_, vQuery_, learningRate, t_, weightDecay_, static_cast<int>(weightCount_));
	++t_;
}
void GlobalPoolLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}
void GlobalPoolLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
void GlobalPoolLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, mQuery_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, vQuery_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}
void GlobalPoolLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(mQuery_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(vQuery_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
size_t GlobalPoolLayer::GetParameterSize(){ return weightCount_*sizeof(__half); }
size_t GlobalPoolLayer::GetOptimizerStateSize(){ return 2*weightCount_*sizeof(__half); }
void GlobalPoolLayer::SetTrain(bool enable){ train_ = enable; }
void GlobalPoolLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!train_) return;
	halfTasks.push_back({weights_, gradQuery_, mQuery_, vQuery_, static_cast<int>(weightCount_), weightDecay_});
}