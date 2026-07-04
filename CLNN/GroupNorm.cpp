#include "GroupNorm.h"
#include "HostCommon.h"
#include "CuCommon.h"
#include <stdexcept>
#include <vector>
GroupNorm::GroupNorm(const int batchSize, const int channels, const int height, const int width, const int groups, std::string layerName, const bool train, const float epsilon)
	: batchSize_(batchSize), outC_(channels), outHW_(height*width), height_(height), width_(width), groups_(groups), epsilon_(epsilon){
	if(batchSize_ <= 0 || outC_ <= 0 || height_ <= 0 || width_ <= 0){ throw std::invalid_argument("GroupNorm dimensions must be positive"); }
	if(groups_ <= 0 || outC_%groups_ != 0){ throw std::invalid_argument("GroupNorm channels must be divisible by groups"); }
	if(epsilon_ <= 0.0f){ throw std::invalid_argument("GroupNorm epsilon must be positive"); }
	channelsPerGroup_ = outC_/groups_;
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = static_cast<size_t>(batchSize_)*outC_*outHW_;
	const size_t statsCount = static_cast<size_t>(batchSize_)*groups_;
	const auto paramSizeBytes = outC_*sizeof(float);
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&gamma_, paramSizeBytes);
	CUDAMallocZero(&beta_, paramSizeBytes);
	CUDAMallocZero(&mean_, statsCount*sizeof(float));
	CUDAMallocZero(&invStd_, statsCount*sizeof(float));
	const std::vector<float> gammaInit(outC_, 1.0f);
	checkCUDA(cudaMemcpy(gamma_, gammaInit.data(), paramSizeBytes, cudaMemcpyHostToDevice));
	if(train_){
		workspaceSize_ = GroupNormBackwardWorkspaceSize(batchSize_, groups_);
		CUDAMallocZero(&workspace_, workspaceSize_);
		CUDAMallocZero(&outGrad_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&gradGamma_, paramSizeBytes);
		CUDAMallocZero(&gradBeta_, paramSizeBytes);
		CUDAMallocZero(&mGamma_, paramSizeBytes);
		CUDAMallocZero(&vGamma_, paramSizeBytes);
		CUDAMallocZero(&mBeta_, paramSizeBytes);
		CUDAMallocZero(&vBeta_, paramSizeBytes);
		trainingAllocated_ = true;
	}
}
GroupNorm::~GroupNorm(){
	cudaFree(outData_);
	cudaFree(gamma_);
	cudaFree(beta_);
	cudaFree(mean_);
	cudaFree(invStd_);
	if(trainingAllocated_){
		cudaFree(outGrad_);
		cudaFree(workspace_);
		cudaFree(gradGamma_);
		cudaFree(gradBeta_);
		cudaFree(mGamma_);
		cudaFree(vGamma_);
		cudaFree(mBeta_);
		cudaFree(vBeta_);
	}
}
__half* GroupNorm::Forward(__half* data){
	inData_ = data;
	GroupNormForward(outData_, data, gamma_, beta_, mean_, invStd_, batchSize_, outC_, outHW_, groups_, epsilon_);
	return outData_;
}
__half* GroupNorm::Backward(__half* grad){
	if(!train_ || !trainingAllocated_){ throw std::runtime_error("GroupNorm::Backward requires training mode"); }
	GroupNormBackward(outGrad_, grad, inData_, gamma_, gradGamma_, gradBeta_, mean_, invStd_, workspace_, workspaceSize_, batchSize_, outC_, outHW_, groups_);
	return outGrad_;
}
void GroupNorm::UpdateParameters(const float learningRate){
	if(!train_ || !trainingAllocated_) return;
	AdamWFloat(gamma_, gradGamma_, mGamma_, vGamma_, learningRate, t_, 0.0f, outC_);
	AdamWFloat(beta_, gradBeta_, mBeta_, vBeta_, learningRate, t_, 0.0f, outC_);
	++t_;
}
void GroupNorm::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, gamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, beta_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}
void GroupNorm::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(gamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(beta_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
void GroupNorm::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!trainingAllocated_) return;
	cudaMemcpy(buffer, mGamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, vGamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, mBeta_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, vBeta_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	file.write(reinterpret_cast<char*>(&t_), sizeof(int));
}
void GroupNorm::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!trainingAllocated_) return;
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(mGamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(vGamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(mBeta_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(vBeta_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t GroupNorm::GetParameterSize(){
	return 2*outC_*sizeof(float);
}
size_t GroupNorm::GetOptimizerStateSize(){
	return 4*outC_*sizeof(float);
}
void GroupNorm::SetTrain(const bool enable){
	if(enable && !trainingAllocated_){ throw std::runtime_error("GroupNorm cannot enable training when constructed for inference"); }
	train_ = enable;
}
void GroupNorm::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!train_ || !trainingAllocated_) return;
	floatTasks.push_back({gamma_, gradGamma_, mGamma_, vGamma_, outC_, 0.0f});
	floatTasks.push_back({beta_, gradBeta_, mBeta_, vBeta_, outC_, 0.0f});
}
