#include "FCLayer.h"
#include "common.h"
#include <iostream>
FCLayer::FCLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int bitchSize, int inputSize, int outputSize, const char* layerName) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle),
																																			bitchSize_(bitchSize), inC_(inputSize), outC_(outputSize), inData_(nullptr),
																																			epsilon_(1e-6){
	layerName_ = layerName;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bitchSize_, inC_, 1, 1));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bitchSize_, outC_, 1, 1));
	outSize_ = outC_*bitchSize_;
	gradOutSize_ = inC_*bitchSize_*sizeof(__half);
	weightCount_ = inC_*outC_;
	checkCUDA(cudaMalloc(&outData_, outSize_*sizeof(__half)));
	checkCUDA(cudaMalloc(&weights_, weightCount_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradWeights_, weightCount_*sizeof(__half)));
	checkCUDA(cudaMalloc(&bias_, outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradBias_, outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradOut_, gradOutSize_));
	checkCUDA(cudaMalloc(&m_weights_, weightCount_*sizeof(float)));
	checkCUDA(cudaMalloc(&v_weights_, weightCount_*sizeof(float)));
	checkCUDA(cudaMalloc(&m_bias_, outC_*sizeof(float)));
	checkCUDA(cudaMalloc(&v_bias_, outC_*sizeof(float)));
	checkCUDA(cudaMemset(outData_, 0, outSize_*sizeof(__half)));
	HeInit(weights_, weightCount_, inC_);
	checkCUDA(cudaMemset(gradWeights_, 0, weightCount_*sizeof(__half)));
	checkCUDA(cudaMemset(bias_, 0, outC_*sizeof(__half)));
	checkCUDA(cudaMemset(gradBias_, 0, outC_*sizeof(__half)));
	checkCUDA(cudaMemset(gradOut_, 0, gradOutSize_));
	checkCUDA(cudaMemset(m_weights_, 0, weightCount_*sizeof(float)));
	checkCUDA(cudaMemset(v_weights_, 0, weightCount_*sizeof(float)));
	checkCUDA(cudaMemset(m_bias_, 0, outC_*sizeof(float)));
	checkCUDA(cudaMemset(v_bias_, 0, outC_*sizeof(float)));
}
FCLayer::~FCLayer(){
	cudaFree(outData_);
	cudaFree(weights_);
	cudaFree(gradOut_);
	cudaFree(gradWeights_);
	cudaFree(bias_);
	cudaFree(gradBias_);
	cudnnDestroyTensorDescriptor(inDesc_);
	cudnnDestroyTensorDescriptor(outDesc_);
	cudaFree(m_weights_);
	cudaFree(v_weights_);
	cudaFree(m_bias_);
	cudaFree(v_bias_);
}
__half* FCLayer::forward(__half* data){
	inData_ = data;
	checkCUBLAS(cublasSgemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, outC_, bitchSize_, inC_, &alpha, weights_, CUDA_R_16F, outC_, data, CUDA_R_16F, inC_, &beta1, outData_, CUDA_R_16F, outC_));
	//std::cout << layerName_ << " ";
	//printDataHalf(outData_, 10, "outData_");
	return outData_;
}
__half* FCLayer::backward(__half* grad){
	//clipGrads(grad, outC_*bitchSize_);
	checkCUBLAS(cublasSgemmEx( cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, inC_, outC_, bitchSize_, &alpha, inData_, CUDA_R_16F, bitchSize_, grad, CUDA_R_16F, bitchSize_, &beta0, gradWeights_, CUDA_R_16F, inC_ ));
	biasGradient(grad, gradBias_, outC_, bitchSize_);
	checkCUBLAS(cublasSgemmEx( cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, inC_, bitchSize_, outC_, &alpha, weights_, CUDA_R_16F, inC_, grad, CUDA_R_16F, bitchSize_, &beta0, gradOut_, CUDA_R_16F, inC_ ));
	std::cout << layerName_ << " ";
	printDataHalf(gradOut_, 10, "gradOut_");
	return gradOut_;
}
void FCLayer::updateParameters(float learningRate){
	//SGDHalf(weights_, learningRate, gradWeights_, weightCount_);
	//SGDHalf(bias_, learningRate, gradBias_, outC_);
	AdamWHalf(weights_, m_weights_, v_weights_, learningRate, gradWeights_, weightCount_, t_, 0.001F);
	AdamWHalf(bias_, m_bias_, v_bias_, learningRate, gradBias_, outC_, t_, 0.001F);
	cudaMemcpy(outData_, bias_, outC_*sizeof(__half), cudaMemcpyDeviceToDevice);
	++t_;
}