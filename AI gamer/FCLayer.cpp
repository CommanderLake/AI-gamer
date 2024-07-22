#include "FCLayer.h"
#include "common.h"
#include <iostream>
FCLayer::FCLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int bitchSize, int inputSize, int outputSize, const char* layerName) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle),
																																			bitchSize_(bitchSize), inC_(inputSize), outC_(outputSize), inData_(nullptr),
																																			epsilon_(1e-6){
	layerName_ = layerName;
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bitchSize_, inC_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, bitchSize_, outC_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(biasDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, outC_, 1, 1));
	outNCHW_ = outC_*bitchSize_;
	gradOutSize_ = inC_*bitchSize_*sizeof(__half);
	weightCount_ = inC_*outC_;
	checkCUDA(cudaMalloc(&outData_, outNCHW_*sizeof(__half)));
	checkCUDA(cudaMalloc(&weights_, weightCount_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradWeights_, weightCount_*sizeof(__half)));
	checkCUDA(cudaMalloc(&bias_, outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradBias_, outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradOut_, gradOutSize_));
	checkCUDA(cudaMalloc(&m_Weights_, weightCount_*sizeof(float)));
	checkCUDA(cudaMalloc(&v_Weights_, weightCount_*sizeof(float)));
	checkCUDA(cudaMalloc(&m_Bias_, outC_*sizeof(float)));
	checkCUDA(cudaMalloc(&v_Bias_, outC_*sizeof(float)));
	checkCUDA(cudaMemset(outData_, 0, outNCHW_*sizeof(__half)));
	HeInit(weights_, weightCount_, inC_);
	checkCUDA(cudaMemset(gradWeights_, 0, weightCount_*sizeof(__half)));
	checkCUDA(cudaMemset(bias_, 0, outC_*sizeof(__half)));
	checkCUDA(cudaMemset(gradBias_, 0, outC_*sizeof(__half)));
	checkCUDA(cudaMemset(gradOut_, 0, gradOutSize_));
	checkCUDA(cudaMemset(m_Weights_, 0, weightCount_*sizeof(float)));
	checkCUDA(cudaMemset(v_Weights_, 0, weightCount_*sizeof(float)));
	checkCUDA(cudaMemset(m_Bias_, 0, outC_*sizeof(float)));
	checkCUDA(cudaMemset(v_Bias_, 0, outC_*sizeof(float)));
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
	cudaFree(m_Weights_);
	cudaFree(v_Weights_);
	cudaFree(m_Bias_);
	cudaFree(v_Bias_);
}
__half* FCLayer::Forward(__half* data){
	inData_ = data;
	checkCUBLAS(cublasSgemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, outC_, bitchSize_, inC_, &alpha, weights_, CUDA_R_16F, outC_, data, CUDA_R_16F, inC_, &beta0, outData_, CUDA_R_16F, outC_));
	cudnnAddTensor(cudnnHandle_, &alpha, biasDesc_, bias_, &beta1, outDesc_, outData_);
	return outData_;
}
__half* FCLayer::Backward(__half* grad){
	//clipGrads(grad, outC_*bitchSize_);
	checkCUBLAS(cublasSgemmEx( cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, inC_, outC_, bitchSize_, &alpha, inData_, CUDA_R_16F, bitchSize_, grad, CUDA_R_16F, bitchSize_, &beta0, gradWeights_, CUDA_R_16F, inC_ ));
	BiasGradient(grad, gradBias_, outC_, bitchSize_);
	cudaDeviceSynchronize();
	if(const cudaError_t err = cudaGetLastError(); err != cudaSuccess){
		printf("CUDA error in Backward: %s\n", cudaGetErrorString(err));
	}
	checkCUBLAS(cublasSgemmEx( cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, inC_, bitchSize_, outC_, &alpha, weights_, CUDA_R_16F, inC_, grad, CUDA_R_16F, bitchSize_, &beta0, gradOut_, CUDA_R_16F, inC_ ));
	return gradOut_;
}
void FCLayer::UpdateParameters(float learningRate){
	//SGDHalf(weights_, learningRate, gradWeights_, weightCount_);
	//SGDHalf(bias_, learningRate, gradBias_, outC_);
	AdamWHalf(weights_, m_Weights_, v_Weights_, learningRate, gradWeights_, weightCount_, t_, 0.001F);
	AdamWHalf(bias_, m_Bias_, v_Bias_, learningRate, gradBias_, outC_, t_, 0.001F);
	++t_;
}