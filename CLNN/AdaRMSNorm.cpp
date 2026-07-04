#include "AdaRMSNorm.h"
#include "CuCommon.h"
#include <stdexcept>
#include <vector>
AdaRMSNorm::AdaRMSNorm(const int batchSize, const int channels, const int height, const int width, std::string layerName, const bool train, const float epsilon)
	: batchSize_(batchSize), outC_(channels), outHW_(height*width), height_(height), width_(width), epsilon_(epsilon){
	if(train){ throw std::invalid_argument("AdaRMSNorm training is not implemented yet"); }
	if(batchSize_ <= 0 || outC_ <= 0 || height_ <= 0 || width_ <= 0){ throw std::invalid_argument("AdaRMSNorm dimensions must be positive"); }
	if(epsilon_ <= 0.0f){ throw std::invalid_argument("AdaRMSNorm epsilon must be positive"); }
	layerName_ = layerName;
	train_ = false;
	outNCHW_ = static_cast<size_t>(batchSize_)*outC_*outHW_;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&gamma_, outC_*sizeof(float));
	CUDAMallocZero(&invRms_, static_cast<size_t>(batchSize_)*sizeof(float));
	const std::vector<float> gammaInit(outC_, 1.0f);
	checkCUDA(cudaMemcpy(gamma_, gammaInit.data(), outC_*sizeof(float), cudaMemcpyHostToDevice));
}
AdaRMSNorm::~AdaRMSNorm(){
	cudaFree(outData_);
	cudaFree(gamma_);
	cudaFree(invRms_);
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
void AdaRMSNorm::ClearModulation(){
	scale_ = nullptr;
	shift_ = nullptr;
	gate_ = nullptr;
	modulationBatchSize_ = 0;
	rowsPerModulation_ = 1;
}
__half* AdaRMSNorm::Forward(__half* data){
	AdaRMSNormForward(outData_, data, gamma_, scale_, shift_, gate_, invRms_, batchSize_, outC_, outHW_, modulationBatchSize_, rowsPerModulation_, epsilon_);
	return outData_;
}
__half* AdaRMSNorm::Backward(__half* grad){
	return grad;
}
void AdaRMSNorm::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, gamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}
void AdaRMSNorm::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(gamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
size_t AdaRMSNorm::GetParameterSize(){
	return outC_*sizeof(float);
}
size_t AdaRMSNorm::GetOptimizerStateSize(){
	return 0;
}
void AdaRMSNorm::SetTrain(const bool enable){
	if(enable){ throw std::runtime_error("AdaRMSNorm training is not implemented yet"); }
	train_ = false;
}
