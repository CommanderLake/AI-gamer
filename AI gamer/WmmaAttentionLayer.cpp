#include "WmmaAttentionLayer.h"
#include "common.h"
#include "CuCommon.cuh"
WmmaAttentionLayer::WmmaAttentionLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int numHeads, const char* layerName, bool train, float weightDecay, const int gradAccumLength, WeightInitMethod weightInitMethod) : cudnnHandle_(cudnnHandle),
	cublasHandle_(cublasHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), numHeads_(numHeads), gradAccumLength_(gradAccumLength), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	headDim_ = embedDim_ / numHeads_;
	outNCHW_ = batchSize_*tokens_*embedDim_;
	alphaWeights_ = 1.0f / (batchSize_*tokens_);
	const size_t projSize = embedDim_*embedDim_;
	CUDAMallocZero(&qWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&kWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&vWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&oWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&workspace_, 4*outNCHW_*sizeof(__half) + batchSize_*tokens_*tokens_*numHeads_*sizeof(float));
	if(train_){
		WeightInit(qWeights_, projSize, embedDim_, weightInitMethod);
		WeightInit(kWeights_, projSize, embedDim_, weightInitMethod);
		WeightInit(vWeights_, projSize, embedDim_, weightInitMethod);
		WeightInit(oWeights_, projSize, embedDim_, weightInitMethod);
		CUDAMallocZero(&gradQ_, projSize*sizeof(__half));
		CUDAMallocZero(&gradK_, projSize*sizeof(__half));
		CUDAMallocZero(&gradV_, projSize*sizeof(__half));
		CUDAMallocZero(&gradOut_, projSize*sizeof(__half));
		CUDAMallocZero(&m_Q_, projSize*sizeof(__half));
		CUDAMallocZero(&v_Q_, projSize*sizeof(__half));
		CUDAMallocZero(&m_K_, projSize*sizeof(__half));
		CUDAMallocZero(&v_K_, projSize*sizeof(__half));
		CUDAMallocZero(&m_V_, projSize*sizeof(__half));
		CUDAMallocZero(&v_V_, projSize*sizeof(__half));
		CUDAMallocZero(&m_O_, projSize*sizeof(__half));
		CUDAMallocZero(&v_O_, projSize*sizeof(__half));
		CUDAMallocZero(&dQ, outNCHW_*sizeof(__half));
		CUDAMallocZero(&dK, outNCHW_*sizeof(__half));
		CUDAMallocZero(&dV, outNCHW_*sizeof(__half));
		CUDAMallocZero(&outGrad_, outNCHW_*sizeof(__half));
	}
}
WmmaAttentionLayer::~WmmaAttentionLayer(){
	cudaFree(qWeights_);
	cudaFree(kWeights_);
	cudaFree(vWeights_);
	cudaFree(oWeights_);
	cudaFree(outData_);
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
		cudaFree(dQ);
		cudaFree(dK);
		cudaFree(dV);
		cudaFree(outGrad_);
	}
}
__half* WmmaAttentionLayer::Forward(__half* data){
	__half* Q = workspace_;
	__half* K = workspace_ + outNCHW_;
	__half* V = workspace_ + 2*outNCHW_;
	__half* attnOut = workspace_ + 3*outNCHW_;
	const auto attentionWeights = reinterpret_cast<float*>(workspace_ + 4*outNCHW_);
	inData_ = data;
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &alpha_, qWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &beta0_, Q, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &alpha_, kWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &beta0_, K, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &alpha_, vWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &beta0_, V, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	WmmaAttention(Q, K, V, attnOut, attentionWeights, batchSize_, tokens_, headDim_, numHeads_);
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &alpha_, oWeights_, CUDA_R_16F, embedDim_, attnOut, CUDA_R_16F, embedDim_, &beta0_, outData_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	return outData_;
}
__half* WmmaAttentionLayer::Backward(__half* grad){
	const __half* Q = workspace_;
	const __half* K = workspace_ + outNCHW_;
	const __half* V = workspace_ + 2*outNCHW_;
	__half* attnOut = workspace_ + 3*outNCHW_;
	const auto attentionWeights = reinterpret_cast<float*>(workspace_ + 4*outNCHW_);
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, grad, CUDA_R_16F, embedDim_, attnOut, CUDA_R_16F, embedDim_, &beta1_, gradOut_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &alpha_, oWeights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &beta0_, attnOut, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	OptimizedWmmaAttentionBackward(Q, K, V, attnOut, attentionWeights, dQ, dK, dV, batchSize_, tokens_, headDim_, numHeads_);
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, dQ, CUDA_R_16F, embedDim_, inData_, CUDA_R_16F, embedDim_, &beta1_, gradQ_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, dK, CUDA_R_16F, embedDim_, inData_, CUDA_R_16F, embedDim_, &beta1_, gradK_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, dV, CUDA_R_16F, embedDim_, inData_, CUDA_R_16F, embedDim_, &beta1_, gradV_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &alpha_, qWeights_, CUDA_R_16F, embedDim_, dQ, CUDA_R_16F, embedDim_, &beta0_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &alpha_, kWeights_, CUDA_R_16F, embedDim_, dK, CUDA_R_16F, embedDim_, &beta1_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &alpha_, vWeights_, CUDA_R_16F, embedDim_, dV, CUDA_R_16F, embedDim_, &beta1_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &alpha_, oWeights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &beta1_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	return outGrad_;
}
void WmmaAttentionLayer::UpdateParameters(float lr){
	AdamWHalf(qWeights_, gradQ_, m_Q_, v_Q_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(kWeights_, gradK_, m_K_, v_K_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(vWeights_, gradV_, m_V_, v_V_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(oWeights_, gradOut_, m_O_, v_O_, lr, t_, weightDecay_, embedDim_*embedDim_);
	++t_;
	cudaMemset(gradQ_, 0, embedDim_*embedDim_*sizeof(__half));
	cudaMemset(gradK_, 0, embedDim_*embedDim_*sizeof(__half));
	cudaMemset(gradV_, 0, embedDim_*embedDim_*sizeof(__half));
	cudaMemset(gradOut_, 0, embedDim_*embedDim_*sizeof(__half));
}
void WmmaAttentionLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	const size_t paramSize = embedDim_*embedDim_*sizeof(__half);
	cudaMemcpy(buffer, qWeights_, paramSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(buffer, kWeights_, paramSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(buffer, vWeights_, paramSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(buffer, oWeights_, paramSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), paramSize);
}
void WmmaAttentionLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	const size_t paramSize = embedDim_*embedDim_*sizeof(__half);
	file.read(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(qWeights_, buffer, paramSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(kWeights_, buffer, paramSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(vWeights_, buffer, paramSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(oWeights_, buffer, paramSize, cudaMemcpyHostToDevice);
}
void WmmaAttentionLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!train_) return;
	const size_t stateSize = embedDim_*embedDim_*sizeof(__half);
	cudaMemcpy(buffer, m_Q_, stateSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, m_K_, stateSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, m_V_, stateSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, m_O_, stateSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, v_Q_, stateSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, v_K_, stateSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, v_V_, stateSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, v_O_, stateSize, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), stateSize);
	file.write(reinterpret_cast<char*>(&t_), sizeof(int));
}
void WmmaAttentionLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!train_) return;
	const size_t stateSize = embedDim_*embedDim_*sizeof(__half);
	file.read(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(m_Q_, buffer, stateSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(m_K_, buffer, stateSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(m_V_, buffer, stateSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(m_O_, buffer, stateSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(v_Q_, buffer, stateSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(v_K_, buffer, stateSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(v_V_, buffer, stateSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(v_O_, buffer, stateSize, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t WmmaAttentionLayer::GetParameterSize(){ return 4*embedDim_*embedDim_*sizeof(__half); }
size_t WmmaAttentionLayer::GetOptimizerStateSize(){
	if(!train_) return 0;
	return 8*embedDim_*embedDim_*sizeof(__half) + sizeof(int);
}
void WmmaAttentionLayer::SetTrain(bool enable){ train_ = enable; }