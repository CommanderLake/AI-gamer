#include "FCLayer.h"
#include "common.h"
#include <iostream>
FCLayer::FCLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inputSize, int outputSize, const char* layerName, bool train) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), batchSize_(batchSize),
																																				inC_(inputSize), outC_(outputSize), inData_(nullptr){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*outC_;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(biasDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, outC_, 1, 1));
	checkCUDA(cudaMalloc(&weights_, inC_*outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&bias_, outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&outData_, outNCHW_*sizeof(__half)));
	HeInit(weights_, inC_*outC_, inC_);
	checkCUDA(cudaMemset(bias_, 0, outC_*sizeof(__half)));
	if(train_){
		checkCUDA(cudaMalloc(&gradIn_, batchSize_*inC_*sizeof(__half)));
		checkCUDA(cudaMalloc(&gradWeights_, inC_*outC_*sizeof(__half)));
		checkCUDA(cudaMalloc(&gradBias_, outC_*sizeof(__half)));
		checkCUDA(cudaMalloc(&m_Weights_, inC_*outC_*sizeof(float)));
		checkCUDA(cudaMalloc(&v_Weights_, inC_*outC_*sizeof(float)));
		checkCUDA(cudaMalloc(&m_Bias_, outC_*sizeof(float)));
		checkCUDA(cudaMalloc(&v_Bias_, outC_*sizeof(float)));
		checkCUDA(cudaMemset(gradWeights_, 0, inC_*outC_*sizeof(__half)));
		checkCUDA(cudaMemset(gradBias_, 0, outC_*sizeof(__half)));
		checkCUDA(cudaMemset(m_Weights_, 0, inC_*outC_*sizeof(float)));
		checkCUDA(cudaMemset(v_Weights_, 0, inC_*outC_*sizeof(float)));
		checkCUDA(cudaMemset(m_Bias_, 0, outC_*sizeof(float)));
		checkCUDA(cudaMemset(v_Bias_, 0, outC_*sizeof(float)));
	}
}
FCLayer::~FCLayer(){
	cudaFree(weights_);
	cudaFree(bias_);
	cudaFree(outData_);
	checkCUDNN(cudnnDestroyTensorDescriptor(inDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasDesc_));
	if(train_){
		cudaFree(gradIn_);
		cudaFree(gradWeights_);
		cudaFree(gradBias_);
		cudaFree(m_Weights_);
		cudaFree(v_Weights_);
		cudaFree(m_Bias_);
		cudaFree(v_Bias_);
	}
}
__half* FCLayer::Forward(__half* data, bool train){
	inData_ = data;
	// Perform matrix multiplication: outData = inData*weights + bias
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, outC_, batchSize_, inC_, &alpha, weights_, CUDA_R_16F, outC_, data, CUDA_R_16F, inC_, &beta0, outData_, CUDA_R_16F, outC_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	// Add bias
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, biasDesc_, bias_, &beta1, outDesc_, outData_));
	return outData_;
}
__half* FCLayer::Backward(__half* grad){
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, outC_, inC_, batchSize_, &alpha, grad, CUDA_R_16F, outC_, inData_, CUDA_R_16F, inC_, &beta0, gradWeights_, CUDA_R_16F, outC_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	BiasGradient(grad, gradBias_, outC_, batchSize_);
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, inC_, batchSize_, outC_, &alpha, weights_, CUDA_R_16F, outC_, grad, CUDA_R_16F, outC_, &beta0, gradIn_, CUDA_R_16F, inC_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	return gradIn_;
}
void FCLayer::UpdateParameters(float learningRate){
	//SGDHalf(weights_, learningRate, gradWeights_, inC_*outC_);
	//SGDHalf(bias_, learningRate, gradBias_, outC_);
	AdamWHalf(weights_, m_Weights_, v_Weights_, learningRate, gradWeights_, inC_*outC_, t_, 0.0001F);
	AdamWHalf(bias_, m_Bias_, v_Bias_, learningRate, gradBias_, outC_, t_, 0.0001F);
	++t_;
}
void FCLayer::SaveParameters(std::ofstream& file, float* buffer) const{
	cudaMemcpy(buffer, weights_, inC_*outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), inC_*outC_*sizeof(__half));
	cudaMemcpy(buffer, bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
}
void FCLayer::LoadParameters(std::ifstream& file, float* buffer){
	file.read(reinterpret_cast<char*>(buffer), inC_*outC_*sizeof(__half));
	cudaMemcpy(weights_, buffer, inC_*outC_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
}
void FCLayer::SaveOptimizerState(std::ofstream& file, float* buffer) const{
	cudaMemcpy(buffer, m_Weights_, inC_*outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), inC_*outC_*sizeof(float));
	cudaMemcpy(buffer, v_Weights_, inC_*outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), inC_*outC_*sizeof(float));
	cudaMemcpy(buffer, m_Bias_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(buffer, v_Bias_, outC_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(float));
}
void FCLayer::LoadOptimizerState(std::ifstream& file, float* buffer){
	file.read(reinterpret_cast<char*>(buffer), inC_*outC_*sizeof(float));
	cudaMemcpy(m_Weights_, buffer, inC_*outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), inC_*outC_*sizeof(float));
	cudaMemcpy(v_Weights_, buffer, inC_*outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(m_Bias_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(float));
	cudaMemcpy(v_Bias_, buffer, outC_*sizeof(float), cudaMemcpyHostToDevice);
}
bool FCLayer::HasParameters() const{
	return true;
}
bool FCLayer::HasOptimizerState() const{
	return true;
}
size_t FCLayer::GetParameterSize() const{
	return inC_*outC_*sizeof(__half);
}
size_t FCLayer::GetOptimizerStateSize() const{
	return inC_*outC_*sizeof(float);
}