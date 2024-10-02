#include "LayerNorm.h"
#include "common.h"
#include <vector>
LayerNorm::LayerNorm(int batchSize, int channels, int height, int width, const char* layerName, float weightDecay) : batchSize_(batchSize), outC_(channels), outHW_(height*width), inData_(nullptr), weightDecay_(weightDecay){
	layerName_ = layerName;
	outCHW_ = outC_*outHW_;
	outNCHW_ = batchSize_*outCHW_;
	const auto paramSizeBytes = outC_*sizeof(float);
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&gamma_, paramSizeBytes);
	CUDAMallocZero(&beta_, paramSizeBytes);
	CUDAMallocZero(&gradGamma_, paramSizeBytes);
	CUDAMallocZero(&gradBeta_, paramSizeBytes);
	CUDAMallocZero(&mean_, paramSizeBytes);
	CUDAMallocZero(&variance_, paramSizeBytes);
	CUDAMallocZero(&mGamma_, paramSizeBytes);
	CUDAMallocZero(&vGamma_, paramSizeBytes);
	CUDAMallocZero(&mBeta_, paramSizeBytes);
	CUDAMallocZero(&vBeta_, paramSizeBytes);
	const std::vector<float> gammaInit(outC_, 1.0f);
	checkCUDA(cudaMemcpy(gamma_, gammaInit.data(), paramSizeBytes, cudaMemcpyHostToDevice));
}
LayerNorm::~LayerNorm(){
	cudaFree(outData_);
	cudaFree(gamma_);
	cudaFree(beta_);
	cudaFree(gradGamma_);
	cudaFree(gradBeta_);
	cudaFree(mean_);
	cudaFree(variance_);
	cudaFree(mGamma_);
	cudaFree(vGamma_);
	cudaFree(mBeta_);
	cudaFree(vBeta_);
}
__half* LayerNorm::Forward(__half* data){
	inData_ = data;
	LayerNormForward(outData_, data, gamma_, beta_, mean_, variance_, batchSize_, outC_, outHW_);
	return outData_;
}
__half* LayerNorm::Backward(__half* grad){
	LayerNormBackward(grad, inData_, gamma_, gradGamma_, gradBeta_, mean_, variance_, batchSize_, outC_, outHW_);
	return grad;
}
void LayerNorm::UpdateParameters(float learningRate){
	AdamWFloat(gamma_, gradGamma_, mGamma_, vGamma_, learningRate, t_, weightDecay_, outC_);
	AdamWFloat(beta_, gradBeta_, mBeta_, vBeta_, learningRate, t_, weightDecay_, outC_);
	++t_;
}
void LayerNorm::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, gamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, beta_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}
void LayerNorm::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(gamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(beta_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
void LayerNorm::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, mGamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, vGamma_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, mBeta_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, vBeta_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}
void LayerNorm::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(mGamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(vGamma_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(mBeta_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(vBeta_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
size_t LayerNorm::GetParameterSize(){
	return outC_*sizeof(float);
}
size_t LayerNorm::GetOptimizerStateSize(){
	return outC_*sizeof(float);
}