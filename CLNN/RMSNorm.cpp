#include "RMSNorm.h"
#include "CuCommon.h"
#include <stdexcept>
#include <vector>
RMSNorm::RMSNorm(const int batchSize, const int channels, const int height, const int width, std::string layerName, const bool train, const bool spatialMode, const float epsilon)
	: batchSize_(batchSize), outC_(channels), outHW_(height*width), height_(height), width_(width), spatialMode_(spatialMode), epsilon_(epsilon){
	if(batchSize_ <= 0 || outC_ <= 0 || height_ <= 0 || width_ <= 0){ throw std::invalid_argument("RMSNorm dimensions must be positive"); }
	if(epsilon_ <= 0.0f){ throw std::invalid_argument("RMSNorm epsilon must be positive"); }
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = static_cast<size_t>(batchSize_)*outC_*outHW_;
	normSize_ = spatialMode_ ? batchSize_*outHW_ : batchSize_;
	const auto paramSizeBytes = outC_*sizeof(float);
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&gamma_, paramSizeBytes);
	CUDAMallocZero(&invRms_, static_cast<size_t>(normSize_)*sizeof(float));
	const std::vector<float> gammaInit(outC_, 1.0f);
	checkCUDA(cudaMemcpy(gamma_, gammaInit.data(), paramSizeBytes, cudaMemcpyHostToDevice));
	if(train_){
		workspaceSize_ = RMSNormBackwardWorkspaceSize(batchSize_, outHW_, spatialMode_);
		CUDAMallocZero(&workspace_, workspaceSize_);
		CUDAMallocZero(&outGrad_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&gradGamma_, paramSizeBytes);
		CUDAMallocZero(&mGamma_, paramSizeBytes);
		CUDAMallocZero(&vGamma_, paramSizeBytes);
		trainingAllocated_ = true;
	}
}
RMSNorm::~RMSNorm(){
	cudaFree(outData_);
	cudaFree(gamma_);
	cudaFree(invRms_);
	if(trainingAllocated_){
		cudaFree(workspace_);
		cudaFree(outGrad_);
		cudaFree(gradGamma_);
		cudaFree(mGamma_);
		cudaFree(vGamma_);
	}
}
__half* RMSNorm::Forward(__half* data){
	if(data == nullptr){ throw std::invalid_argument("RMSNorm::Forward received null input"); }
	inData_ = data;
	RMSNormForward(outData_, data, gamma_, invRms_, batchSize_, outC_, outHW_, spatialMode_, epsilon_);
	return outData_;
}
__half* RMSNorm::Backward(__half* grad){
	if(grad == nullptr){ throw std::invalid_argument("RMSNorm::Backward received null gradient"); }
	if(!train_){ return grad; }
	if(!trainingAllocated_){ throw std::runtime_error("RMSNorm::Backward requires training buffers; construct with train=true"); }
	if(inData_ == nullptr){ throw std::runtime_error("RMSNorm::Backward requires a previous Forward call"); }
	RMSNormBackward(outGrad_, grad, inData_, gamma_, gradGamma_, invRms_, workspace_, workspaceSize_, batchSize_, outC_, outHW_, spatialMode_);
	return outGrad_;
}
void RMSNorm::UpdateParameters(const float learningRate){
	if(!train_ || !trainingAllocated_) return;
	AdamWFloat(gamma_, gradGamma_, mGamma_, vGamma_, learningRate, t_, 0.0f, outC_);
	++t_;
}
void RMSNorm::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, gamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}
void RMSNorm::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(gamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
void RMSNorm::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!trainingAllocated_) return;
	cudaMemcpy(buffer, mGamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, vGamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	file.write(reinterpret_cast<char*>(&t_), sizeof(int));
}
void RMSNorm::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!trainingAllocated_) return;
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(mGamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(vGamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t RMSNorm::GetParameterSize(){
	return outC_*sizeof(float);
}
size_t RMSNorm::GetOptimizerStateSize(){
	return trainingAllocated_ ? 2*outC_*sizeof(float) : 0;
}
void RMSNorm::SetTrain(const bool enable){
	if(enable && !trainingAllocated_){ throw std::runtime_error("RMSNorm cannot enable training when constructed for inference"); }
	train_ = enable;
}
void RMSNorm::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!train_ || !trainingAllocated_) return;
	floatTasks.push_back({gamma_, gradGamma_, mGamma_, vGamma_, outC_, 0.0f});
}
