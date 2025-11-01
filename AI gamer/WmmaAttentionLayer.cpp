#include "WmmaAttentionLayer.h"
#include "common.h"
#include "CuCommon.cuh"
WmmaAttentionLayer::WmmaAttentionLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int numHeads, const char* layerName, bool train, float weightDecay, const int gradAccumLength, WeightInitMethod weightInitMethod) : cudnnHandle_(cudnnHandle),
cublasHandle_(cublasHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), numHeads_(numHeads), gradAccumLength_(gradAccumLength), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	headDim_ = embedDim_ / numHeads_;
	outNCHW_ = batchSize_*tokens_*embedDim_;
	alphaWeights_ = 1.0f / (batchSize_*gradAccumLength_);
	const size_t projSize = embedDim_*embedDim_;
	CUDAMallocZero(&qWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&kWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&vWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&oWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	const auto attentionElems = static_cast<size_t>(batchSize_)*tokens_*tokens_*numHeads_;
	CUDAMallocZero(&workspace_, 4*outNCHW_*sizeof(__half) + attentionElems*sizeof(__half));
	CUDAMallocZero(&qPacked_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&kPacked_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&vPacked_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&attnOutPacked_, outNCHW_*sizeof(__half));
	if(train_){
		WeightInit(qWeights_, projSize, embedDim_, embedDim_, weightInitMethod);
		WeightInit(kWeights_, projSize, embedDim_, embedDim_, weightInitMethod);
		WeightInit(vWeights_, projSize, embedDim_, embedDim_, weightInitMethod);
		WeightInit(oWeights_, projSize, embedDim_, embedDim_, weightInitMethod);
		const auto gradWorkspaceElems = static_cast<size_t>(batchSize_)*tokens_*tokens_*numHeads_;
		attnGradWorkspaceSize_ = gradWorkspaceElems;
		if(gradWorkspaceElems > 0){ CUDAMallocZero(&attnGradWorkspace_, gradWorkspaceElems*sizeof(float)); }
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
		CUDAMallocZero(&outGrad_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&dQPacked_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&dKPacked_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&dVPacked_, outNCHW_*sizeof(__half));
	}
}
WmmaAttentionLayer::~WmmaAttentionLayer(){
	cudaFree(qWeights_);
	cudaFree(kWeights_);
	cudaFree(vWeights_);
	cudaFree(oWeights_);
	cudaFree(outData_);
	cudaFree(workspace_);
	cudaFree(qPacked_);
	cudaFree(kPacked_);
	cudaFree(vPacked_);
	cudaFree(attnOutPacked_);
	if(train_){
		cudaFree(attnGradWorkspace_);
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
		cudaFree(outGrad_);
		cudaFree(dQPacked_);
		cudaFree(dKPacked_);
		cudaFree(dVPacked_);
	}
}
__half* WmmaAttentionLayer::Forward(__half* data){
	const auto Q = workspace_;
	const auto K = workspace_ + outNCHW_;
	const auto V = workspace_ + 2*outNCHW_;
	const auto attnOut = workspace_ + 3*outNCHW_;
	const auto attentionWeights = workspace_ + 4*outNCHW_;
	inData_ = data;
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, qWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &zero_, Q, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, kWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &zero_, K, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, vWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &zero_, V, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	PackColumnsToHeads(Q, K, V, qPacked_, kPacked_, vPacked_, batchSize_, tokens_, embedDim_, numHeads_);
	dQ = Q;
	dK = K;
	dV = V;
	WmmaAttention(qPacked_, kPacked_, vPacked_, attnOutPacked_, train_ ? attentionWeights : nullptr, batchSize_, tokens_, headDim_, numHeads_);
	PackHeadsToColumns(attnOutPacked_, attnOut, batchSize_, tokens_, embedDim_, numHeads_);
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, oWeights_, CUDA_R_16F, embedDim_, attnOut, CUDA_R_16F, embedDim_, &zero_, outData_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	return outData_;
}
__half* WmmaAttentionLayer::Backward(__half* grad){
	const auto attnOut = workspace_ + 3*outNCHW_;
	const __half* attentionWeights = workspace_ + 4*outNCHW_;
	const float* betaWeights = accumCount_++ % gradAccumLength_ == 0 ? &zero_ : &one_;
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, grad, CUDA_R_16F, embedDim_, attnOut, CUDA_R_16F, embedDim_, betaWeights, gradOut_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, oWeights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &zero_, attnOut, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	PackColumnsToHeads(attnOut, attnOutPacked_, batchSize_, tokens_, embedDim_, numHeads_);
	WmmaAttentionBackward(qPacked_, kPacked_, vPacked_, attnOutPacked_, attentionWeights, dQPacked_, dKPacked_, dVPacked_, attnGradWorkspace_, attnGradWorkspaceSize_, batchSize_, tokens_, headDim_, numHeads_);
	PackHeadsToColumns(dQPacked_, dKPacked_, dVPacked_, dQ, dK, dV, batchSize_, tokens_, embedDim_, numHeads_);
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, dQ, CUDA_R_16F, embedDim_, inData_, CUDA_R_16F, embedDim_, betaWeights, gradQ_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, dK, CUDA_R_16F, embedDim_, inData_, CUDA_R_16F, embedDim_, betaWeights, gradK_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, dV, CUDA_R_16F, embedDim_, inData_, CUDA_R_16F, embedDim_, betaWeights, gradV_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, qWeights_, CUDA_R_16F, embedDim_, dQ, CUDA_R_16F, embedDim_, &zero_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, kWeights_, CUDA_R_16F, embedDim_, dK, CUDA_R_16F, embedDim_, &one_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublasHandle_, CUBLAS_OP_T, CUBLAS_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, vWeights_, CUDA_R_16F, embedDim_, dV, CUDA_R_16F, embedDim_, &one_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	return outGrad_;
}
void WmmaAttentionLayer::UpdateParameters(float lr){
	if(accumCount_ % gradAccumLength_ > 0) return;
	AdamWHalf(qWeights_, gradQ_, m_Q_, v_Q_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(kWeights_, gradK_, m_K_, v_K_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(vWeights_, gradV_, m_V_, v_V_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(oWeights_, gradOut_, m_O_, v_O_, lr, t_, weightDecay_, embedDim_*embedDim_);
	++t_;
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