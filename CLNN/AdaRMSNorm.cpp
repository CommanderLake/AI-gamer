#include "AdaRMSNorm.h"
#include "CuCommon.h"
#include <stdexcept>
#include <vector>
AdaRMSNorm::AdaRMSNorm(const int batchSize, const int channels, const int height, const int width, std::string layerName, const bool train, const float epsilon)
	: batchSize_(batchSize), outC_(channels), outHW_(height*width), height_(height), width_(width), epsilon_(epsilon){
	if(batchSize_ <= 0 || outC_ <= 0 || height_ <= 0 || width_ <= 0){ throw std::invalid_argument("AdaRMSNorm dimensions must be positive"); }
	if(epsilon_ <= 0.0f){ throw std::invalid_argument("AdaRMSNorm epsilon must be positive"); }
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = static_cast<size_t>(batchSize_)*outC_*outHW_;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&gamma_, outC_*sizeof(float));
	CUDAMallocZero(&invRms_, static_cast<size_t>(batchSize_)*sizeof(float));
	const std::vector<float> gammaInit(outC_, 1.0f);
	checkCUDA(cudaMemcpy(gamma_, gammaInit.data(), outC_*sizeof(float), cudaMemcpyHostToDevice));
	if(train_){
		workspaceSize_ = AdaRMSNormBackwardWorkspaceSize(batchSize_);
		CUDAMallocZero(&workspace_, workspaceSize_);
		CUDAMallocZero(&outGrad_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&gradGamma_, outC_*sizeof(float));
		CUDAMallocZero(&mGamma_, outC_*sizeof(float));
		CUDAMallocZero(&vGamma_, outC_*sizeof(float));
		trainingAllocated_ = true;
	}
}
AdaRMSNorm::~AdaRMSNorm(){
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
void AdaRMSNorm::SetModulation(const __half* scale, const __half* shift, const __half* gate, const int modulationBatchSize, const int rowsPerModulation){
	if(!scale && !shift && !gate){ throw std::invalid_argument("AdaRMSNorm modulation must include at least one tensor"); }
	if(modulationBatchSize <= 0 || rowsPerModulation <= 0){ throw std::invalid_argument("AdaRMSNorm modulation dimensions must be positive"); }
	if(static_cast<size_t>(modulationBatchSize)*rowsPerModulation < static_cast<size_t>(batchSize_)){
		throw std::invalid_argument("AdaRMSNorm modulation does not cover all rows");
	}
	scale_ = scale;
	shift_ = shift;
	gate_ = gate;
	modulationBatchSize_ = modulationBatchSize;
	rowsPerModulation_ = rowsPerModulation;
}
void AdaRMSNorm::SetModulationGradients(__half* gradScale, __half* gradShift, __half* gradGate){
	gradScale_ = gradScale;
	gradShift_ = gradShift;
	gradGate_ = gradGate;
}
void AdaRMSNorm::ClearModulation(){
	scale_ = nullptr;
	shift_ = nullptr;
	gate_ = nullptr;
	gradScale_ = nullptr;
	gradShift_ = nullptr;
	gradGate_ = nullptr;
	modulationBatchSize_ = 0;
	rowsPerModulation_ = 1;
}
__half* AdaRMSNorm::Forward(__half* data){
	if(data == nullptr){ throw std::invalid_argument("AdaRMSNorm::Forward received null input"); }
	inData_ = data;
	AdaRMSNormForward(outData_, data, gamma_, scale_, shift_, gate_, invRms_, batchSize_, outC_, outHW_, modulationBatchSize_, rowsPerModulation_, epsilon_);
	return outData_;
}
__half* AdaRMSNorm::Backward(__half* grad){
	if(grad == nullptr){ throw std::invalid_argument("AdaRMSNorm::Backward received null gradient"); }
	if(!train_){ return grad; }
	if(!trainingAllocated_){ throw std::runtime_error("AdaRMSNorm::Backward requires training buffers; construct with train=true"); }
	if(inData_ == nullptr){ throw std::runtime_error("AdaRMSNorm::Backward requires a previous Forward call"); }
	AdaRMSNormBackward(outGrad_, grad, inData_, gamma_, gradGamma_, invRms_, scale_, shift_, gate_, scale_ ? gradScale_ : nullptr, shift_ ? gradShift_ : nullptr, gate_ ? gradGate_ : nullptr, workspace_, workspaceSize_, batchSize_, outC_, outHW_, modulationBatchSize_, rowsPerModulation_);
	return outGrad_;
}
void AdaRMSNorm::UpdateParameters(const float learningRate){
	if(!train_ || !trainingAllocated_) return;
	AdamWFloat(gamma_, gradGamma_, mGamma_, vGamma_, learningRate, t_, 0.0f, outC_);
	++t_;
}
void AdaRMSNorm::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, gamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}
void AdaRMSNorm::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(gamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
void AdaRMSNorm::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!trainingAllocated_) return;
	cudaMemcpy(buffer, mGamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, vGamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	file.write(reinterpret_cast<const char*>(&t_), sizeof(int));
}
void AdaRMSNorm::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!trainingAllocated_){ return; }
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(mGamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(vGamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t AdaRMSNorm::GetParameterSize(){
	return outC_*sizeof(float);
}
size_t AdaRMSNorm::GetOptimizerStateSize(){
	return trainingAllocated_ ? 2*outC_*sizeof(float) : 0;
}
void AdaRMSNorm::SetTrain(const bool enable){
	if(enable && !trainingAllocated_){ throw std::runtime_error("AdaRMSNorm cannot enable training when constructed for inference"); }
	train_ = enable;
}
void AdaRMSNorm::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!train_ || !trainingAllocated_) return;
	floatTasks.push_back({gamma_, gradGamma_, mGamma_, vGamma_, outC_, 0.0f});
}
__half* AdaRMSNorm::GetScaleGrad(){
	return gradScale_;
}
__half* AdaRMSNorm::GetShiftGrad(){
	return gradShift_;
}
__half* AdaRMSNorm::GetGateGrad(){
	return gradGate_;
}
