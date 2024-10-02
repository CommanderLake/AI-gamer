#include "FCLayer.h"
#include "common.h"
#include <iostream>
FCLayer::FCLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inC, int outC, const char* layerName, bool train, float weightDecay) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle),
	batchSize_(batchSize), inC_(inC), outC_(outC), inData_(nullptr), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*outC_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(biasDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, outC_, 1, 1));
	weightCount_ = inC_*outC_;
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	CUDAMallocZero(&bias_, outC_*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	HeInit(weights_, weightCount_, inC_);
	if(train_){
		CUDAMallocZero(&gradWeights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&gradBias_, outC_*sizeof(__half));
		CUDAMallocZero(&gradOut_, batchSize_*inC_*sizeof(__half));
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&v_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&m_Bias_, outC_*sizeof(__half));
			CUDAMallocZero(&v_Bias_, outC_*sizeof(__half));
		}
	}
}
FCLayer::~FCLayer(){
	cudaFree(weights_);
	cudaFree(bias_);
	cudaFree(outData_);
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasDesc_));
	if(train_){
		cudaFree(gradOut_);
		cudaFree(gradWeights_);
		cudaFree(gradBias_);
		if(useAdamW_){
			cudaFree(m_Weights_);
			cudaFree(v_Weights_);
			cudaFree(m_Bias_);
			cudaFree(v_Bias_);
		}
	}
}
__half* FCLayer::Forward(__half* data){
	inData_ = data;
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, outC_, batchSize_, inC_, &beta1, weights_, CUDA_R_16F, outC_, data, CUDA_R_16F, inC_, &beta0, outData_, CUDA_R_16F, outC_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, biasDesc_, bias_, &beta1, outDesc_, outData_));
	return outData_;
}
__half* FCLayer::Backward(__half* grad){
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, outC_, inC_, batchSize_, &alpha, grad, CUDA_R_16F, outC_, inData_, CUDA_R_16F, inC_, &beta0, gradWeights_, CUDA_R_16F, outC_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	BiasGradient(grad, gradBias_, outC_, batchSize_);
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, inC_, batchSize_, outC_, &alpha, weights_, CUDA_R_16F, outC_, grad, CUDA_R_16F, outC_, &beta0, gradOut_, CUDA_R_16F, inC_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	return gradOut_;
}
void FCLayer::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, weightCount_);
		AdamWHalf(bias_, gradBias_, m_Bias_, v_Bias_, learningRate, t_, weightDecay_, outC_);
		++t_;
	} else{
		SGDHalf(weights_, gradWeights_, weightCount_, learningRate, weightDecay_);
		SGDHalf(bias_, gradBias_, outC_, learningRate, weightDecay_);
	}
}
void FCLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
}
void FCLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
}
void FCLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, m_Bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(buffer, v_Bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
}
void FCLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(m_Bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(v_Bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
}
size_t FCLayer::GetParameterSize(){
	return weightCount_*sizeof(__half);
}
size_t FCLayer::GetOptimizerStateSize(){
	return weightCount_*sizeof(__half);
}