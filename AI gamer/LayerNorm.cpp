#include "LayerNorm.h"
#include "common.h"
#include "CuCommon.cuh"
#include <vector>
LayerNorm::LayerNorm(const int batchSize, const int channels, const int height, const int width, const int tokensPerSample, const char* layerName, const bool train) : ogbs_(batchSize), batchSize_(batchSize), tokenBatchSize_(tokensPerSample), outC_(channels), outHW_(height*width), height_(height), width_(width){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*outC_*outHW_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, height_, width_));
	const auto paramSizeBytes = outC_*sizeof(float);
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&gamma_, paramSizeBytes);
	CUDAMallocZero(&beta_, paramSizeBytes);
	CUDAMallocZero(&mean_, batchSize_*sizeof(float));
	CUDAMallocZero(&variance_, batchSize_*sizeof(float));
	const std::vector<float> gammaInit(outC_, 1.0f);
	checkCUDA(cudaMemcpy(gamma_, gammaInit.data(), paramSizeBytes, cudaMemcpyHostToDevice));
	if(train){
		workspaceSize_ = 2*batchSize_*sizeof(float);
		CUDAMallocZero(&workspace_, workspaceSize_);
		CUDAMallocZero(&outGrad_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&gradGamma_, paramSizeBytes);
		CUDAMallocZero(&gradBeta_, paramSizeBytes);
		CUDAMallocZero(&mGamma_, paramSizeBytes);
		CUDAMallocZero(&vGamma_, paramSizeBytes);
		CUDAMallocZero(&mBeta_, paramSizeBytes);
		CUDAMallocZero(&vBeta_, paramSizeBytes);
	}
}
LayerNorm::~LayerNorm(){
	cudaFree(outData_);
	cudaFree(gamma_);
	cudaFree(beta_);
	cudaFree(mean_);
	cudaFree(variance_);
	if(train_){
		cudaFree(outGrad_);
		cudaFree(workspace_);
		cudaFree(gradGamma_);
		cudaFree(gradBeta_);
		cudaFree(mGamma_);
		cudaFree(vGamma_);
		cudaFree(mBeta_);
		cudaFree(vBeta_);
	}
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
}
__half* LayerNorm::Forward(__half* data){
	inData_ = data;
	LayerNormForward(outData_, data, gamma_, beta_, mean_, variance_, batchSize_, outC_, outHW_);
	return outData_;
}
__half* LayerNorm::Backward(__half* grad){
	LayerNormBackward(outGrad_, grad, inData_, gamma_, gradGamma_, gradBeta_, mean_, variance_, workspace_, workspaceSize_, batchSize_, outC_, outHW_);
	return outGrad_;
}
void LayerNorm::UpdateParameters(float learningRate){
	AdamWFloat(gamma_, gradGamma_, mGamma_, vGamma_, learningRate, t_, 0.0f, outC_);
	AdamWFloat(beta_, gradBeta_, mBeta_, vBeta_, learningRate, t_, 0.0f, outC_);
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
	return 2*outC_*sizeof(float);
}
size_t LayerNorm::GetOptimizerStateSize(){
	return 4*outC_*sizeof(float);
}
void LayerNorm::SetTrain(const bool enable){
	batchSize_ = enable ? ogbs_ : tokenBatchSize_;
	outNCHW_ = batchSize_*outC_*outHW_;
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, height_, width_));
}