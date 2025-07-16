#include "WmmaAttentionLayer.h"
#include "common.h"
#include <iostream>
WmmaAttentionLayer::WmmaAttentionLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int numHeads, const char* layerName, bool train, float weightDecay) : cudnnHandle_(cudnnHandle),
	cublasHandle_(cublasHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), numHeads_(numHeads), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	headDim_ = embedDim_ / numHeads_;
	outNCHW_ = batchSize_*tokens_*embedDim_;
	size_t projSize = embedDim_*embedDim_;
	CUDAMallocZero(&qWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&kWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&vWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&oWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&workspace_, outNCHW_*sizeof(__half));
	if(train_){
		WeightInit(qWeights_, projSize, embedDim_);
		WeightInit(kWeights_, projSize, embedDim_);
		WeightInit(vWeights_, projSize, embedDim_);
		WeightInit(oWeights_, projSize, embedDim_);
		CUDAMallocZero(&gradQ_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&gradK_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&gradV_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&gradOut_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&m_Q_, projSize*sizeof(__half));
		CUDAMallocZero(&v_Q_, projSize*sizeof(__half));
		CUDAMallocZero(&m_K_, projSize*sizeof(__half));
		CUDAMallocZero(&v_K_, projSize*sizeof(__half));
		CUDAMallocZero(&m_V_, projSize*sizeof(__half));
		CUDAMallocZero(&v_V_, projSize*sizeof(__half));
		CUDAMallocZero(&m_O_, projSize*sizeof(__half));
		CUDAMallocZero(&v_O_, projSize*sizeof(__half));
	}
}
WmmaAttentionLayer::~WmmaAttentionLayer(){
	cudaFree(qWeights_);
	cudaFree(kWeights_);
	cudaFree(vWeights_);
	cudaFree(oWeights_);
	cudaFree(workspace_);
	if(train_){
		cudaFree(gradQ_);
		cudaFree(gradK_);
		cudaFree(gradV_);
		cudaFree(gradOut_);
		cudaFree(m_Q_);
		cudaFree(v_Q_);
		cudaFree(m_K_);
		cudaFree(v_K_);
		cudaFree(m_V_);
		cudaFree(v_V_);
		cudaFree(m_O_);
		cudaFree(v_O_);
	}
}
__half* WmmaAttentionLayer::Forward(__half* data){
	// linear projections
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N,
		embedDim_, tokens_*batchSize_, embedDim_, &alpha_, qWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &beta0_, workspace_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	const __half* Q = workspace_;
	const __half* K = workspace_ + batchSize_*tokens_*embedDim_;
	const __half* V = workspace_ + 2*batchSize_*tokens_*embedDim_;
	// project for K
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N,
		embedDim_, tokens_*batchSize_, embedDim_, &alpha_, kWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &beta0_, const_cast<__half*>(K), CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	// project for V
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N,
		embedDim_, tokens_*batchSize_, embedDim_, &alpha_, vWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &beta0_, const_cast<__half*>(V), CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	WmmaAttention(Q, K, V, outData_, batchSize_, tokens_, embedDim_, numHeads_);
	return outData_;
}
__half* WmmaAttentionLayer::Backward(__half* grad){
	// not implemented for brevity
	return gradOut_;
}
void WmmaAttentionLayer::UpdateParameters(float lr){
	AdamWHalf(qWeights_, gradQ_, m_Q_, v_Q_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(kWeights_, gradK_, m_K_, v_K_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(vWeights_, gradV_, m_V_, v_V_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(oWeights_, gradOut_, m_O_, v_O_, lr, t_, weightDecay_, embedDim_*embedDim_);
	++t_;
}
void WmmaAttentionLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, qWeights_, embedDim_*embedDim_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), embedDim_*embedDim_*sizeof(__half));
	cudaMemcpy(buffer, kWeights_, embedDim_*embedDim_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), embedDim_*embedDim_*sizeof(__half));
	cudaMemcpy(buffer, vWeights_, embedDim_*embedDim_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), embedDim_*embedDim_*sizeof(__half));
	cudaMemcpy(buffer, oWeights_, embedDim_*embedDim_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), embedDim_*embedDim_*sizeof(__half));
}
void WmmaAttentionLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), embedDim_*embedDim_*sizeof(__half));
	cudaMemcpy(qWeights_, buffer, embedDim_*embedDim_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), embedDim_*embedDim_*sizeof(__half));
	cudaMemcpy(kWeights_, buffer, embedDim_*embedDim_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), embedDim_*embedDim_*sizeof(__half));
	cudaMemcpy(vWeights_, buffer, embedDim_*embedDim_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), embedDim_*embedDim_*sizeof(__half));
	cudaMemcpy(oWeights_, buffer, embedDim_*embedDim_*sizeof(__half), cudaMemcpyHostToDevice);
}
void WmmaAttentionLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){ }
void WmmaAttentionLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){ }
size_t WmmaAttentionLayer::GetParameterSize(){ return 4*embedDim_*embedDim_*sizeof(__half); }
size_t WmmaAttentionLayer::GetOptimizerStateSize(){ return 0; }
void WmmaAttentionLayer::SetTrain(bool enable){ train_ = enable; }