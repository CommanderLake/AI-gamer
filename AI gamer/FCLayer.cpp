#include "FCLayer.h"
#include "common.h"
#include "CuCommon.cuh"
#include <iostream>
FCLayer::FCLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int inC, const int outC, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength, WeightInitMethod weightInitMethod) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), ogbs_(batchSize), batchSize_(batchSize), inC_(inC), outC_(outC), inData_(nullptr), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*outC_;
	alphaWeights_ = 1.0f/(batchSize_*gradAccumLength_);
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, 1, 1));
	weightCount_ = inC_*outC_;
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	if(train_){
		OrthogonalInit(weights_, inC_, outC_, weightInitMethod);
		CUDAMallocZero(&gradWeights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&outGrad_, batchSize_*inC_*sizeof(__half));
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&v_Weights_, weightCount_*sizeof(__half));
		}
	}
}
FCLayer::~FCLayer(){
	cudaFree(weights_);
	cudaFree(outData_);
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	if(train_){
		cudaFree(outGrad_);
		cudaFree(gradWeights_);
		if(useAdamW_){
			cudaFree(m_Weights_);
			cudaFree(v_Weights_);
		}
	}
}
__half* FCLayer::Forward(__half* data){
	inData_ = data;
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, outC_, batchSize_, inC_, &alpha_, weights_, CUDA_R_16F, outC_, data, CUDA_R_16F, inC_, &beta0_, outData_, CUDA_R_16F, outC_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	return outData_;
}
__half* FCLayer::Backward(__half* grad){
	const float* betaWeights = accumCount_++%gradAccumLength_==0 ? &beta0_ : &beta1_;
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, outC_, inC_, batchSize_, &alphaWeights_, grad, CUDA_R_16F, outC_, inData_, CUDA_R_16F, inC_, betaWeights, gradWeights_, CUDA_R_16F, outC_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, inC_, batchSize_, outC_, &alpha_, weights_, CUDA_R_16F, outC_, grad, CUDA_R_16F, outC_, &beta0_, outGrad_, CUDA_R_16F, inC_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	return outGrad_;
}
void FCLayer::UpdateParameters(const float learningRate){
	if(accumCount_%gradAccumLength_>0) return;
	if(useAdamW_){
		++t_;
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, weightCount_);
	} else{ SGDHalf(weights_, gradWeights_, weightCount_, learningRate, weightDecay_); }
}
void FCLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}
void FCLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
void FCLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}
void FCLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
size_t FCLayer::GetParameterSize(){ return weightCount_*sizeof(__half); }
size_t FCLayer::GetOptimizerStateSize(){ return weightCount_*sizeof(__half); }
void FCLayer::SetTrain(const bool enable){
	if(enable){
		train_ = true;
		batchSize_ = ogbs_;
	} else{
		train_ = false;
		batchSize_ = 1;
	}
	outNCHW_ = batchSize_*outC_;
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, 1, 1));
}