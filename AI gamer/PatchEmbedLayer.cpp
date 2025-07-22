#include "PatchEmbedLayer.h"
#include "common.h"
#include "CuCommon.cuh"
PatchEmbedLayer::PatchEmbedLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inC, int inH, int inW, int patchSize, int embedDim, const char* layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod) :
	cudnn_(cudnnHandle), cublas_(cublasHandle), ogbs_(batchSize), batchSize_(batchSize), inC_(inC), inH_(inH), inW_(inW), patchSize_(patchSize), embedDim_(embedDim), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	patchRows_ = DivCeil(inH_, patchSize_);
	patchCols_ = DivCeil(inW_, patchSize_);
	patchDim_ = inC_*patchSize_*patchSize_;
	numPatches_ = patchRows_*patchCols_;
	outNCHW_ = batchSize_*embedDim_*numPatches_;
	alphaWeights_ = 1.0f / (batchSize_*gradAccumLength_);
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, embedDim_, patchRows_, patchCols_));
	weightCount_ = embedDim_*patchDim_;
	posCount_ = embedDim_*numPatches_;
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	CUDAMallocZero(&posEmbed_, posCount_*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&patchBuffer_, batchSize_*numPatches_*patchDim_*sizeof(__half));
	checkCUDNN(cudnnCreateTensorDescriptor(&posDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(posDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, embedDim_, patchRows_, patchCols_));
	if(train_){
		WeightInit(weights_, weightCount_, embedDim_, weightInitMethod);
		CUDAMallocZero(&gradWeights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&gradPosEmbed_, posCount_*sizeof(__half));
		CUDAMallocZero(&outGrad_, batchSize_*inC_*inH_*inW_*sizeof(__half));
		CUDAMallocZero(&gradPatchBuffer_, batchSize_*numPatches_*patchDim_*sizeof(__half));
		CUDAMallocZero(&m_Weights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&v_Weights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&m_PosEmbed_, posCount_*sizeof(__half));
		CUDAMallocZero(&v_PosEmbed_, posCount_*sizeof(__half));
	}
}
PatchEmbedLayer::~PatchEmbedLayer(){
	cudaFree(weights_);
	cudaFree(posEmbed_);
	cudaFree(outData_);
	cudaFree(patchBuffer_);
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(posDesc_));
	if(train_){
		cudaFree(gradWeights_);
		cudaFree(gradPosEmbed_);
		cudaFree(outGrad_);
		cudaFree(gradPatchBuffer_);
		cudaFree(m_Weights_);
		cudaFree(v_Weights_);
		cudaFree(m_PosEmbed_);
		cudaFree(v_PosEmbed_);
	}
}
__half* PatchEmbedLayer::Forward(__half* data){
	inData_ = data;
	ExtractPatches(data, patchBuffer_, batchSize_, inC_, inH_, inW_, patchSize_);
	checkCUBLAS(cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, batchSize_*numPatches_, patchDim_, &alpha_, weights_, CUDA_R_16F, embedDim_, patchBuffer_, CUDA_R_16F, patchDim_, &beta0_, outData_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUDNN(cudnnAddTensor(cudnn_, &alpha_, posDesc_, posEmbed_, &alpha_, outDesc_, outData_));
	return outData_;
}
__half* PatchEmbedLayer::Backward(__half* grad){
	const float* betaWeights = accumCount_++ % gradAccumLength_ == 0 ? &beta0_ : &beta1_;
	checkCUBLAS(cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, patchDim_, batchSize_*numPatches_, &alphaWeights_, grad, CUDA_R_16F, embedDim_, patchBuffer_, CUDA_R_16F, patchDim_, betaWeights, gradWeights_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, patchDim_, batchSize_*numPatches_, embedDim_, &alpha_, weights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &beta0_, gradPatchBuffer_, CUDA_R_16F, patchDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	const bool zeroPos = ((accumCount_-1) % gradAccumLength_) == 0;
	SumPositionalGrad(grad, gradPosEmbed_, batchSize_, embedDim_, numPatches_, zeroPos);
	CombinePatchGrads(gradPatchBuffer_, outGrad_, batchSize_, inC_, inH_, inW_, patchSize_);
	return outGrad_;
}
void PatchEmbedLayer::UpdateParameters(float lr){
	if(accumCount_ % gradAccumLength_ > 0) return;
	if(useAdamW_){
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, lr, t_, weightDecay_, weightCount_);
		AdamWHalf(posEmbed_, gradPosEmbed_, m_PosEmbed_, v_PosEmbed_, lr, t_, weightDecay_, posCount_);
	} else{
		SGDHalf(weights_, gradWeights_, weightCount_, lr, weightDecay_);
		SGDHalf(posEmbed_, gradPosEmbed_, posCount_, lr, weightDecay_);
	}
	++t_;
}
void PatchEmbedLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, posEmbed_, posCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), posCount_*sizeof(__half));
}
void PatchEmbedLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), posCount_*sizeof(__half));
	cudaMemcpy(posEmbed_, buffer, posCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
void PatchEmbedLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, m_PosEmbed_, posCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), posCount_*sizeof(__half));
	cudaMemcpy(buffer, v_PosEmbed_, posCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), posCount_*sizeof(__half));
	file.write(reinterpret_cast<char*>(&t_), sizeof(int));
}
void PatchEmbedLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), posCount_*sizeof(__half));
	cudaMemcpy(m_PosEmbed_, buffer, posCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), posCount_*sizeof(__half));
	cudaMemcpy(v_PosEmbed_, buffer, posCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t PatchEmbedLayer::GetParameterSize(){ return (weightCount_ + posCount_)*sizeof(__half); }
size_t PatchEmbedLayer::GetOptimizerStateSize(){ return useAdamW_ ? (weightCount_ + posCount_)*sizeof(__half)*2 + sizeof(int) : 0; }
void PatchEmbedLayer::SetTrain(bool enable){
	if(enable){
		train_ = true;
		batchSize_ = ogbs_;
	} else{
		train_ = false;
		batchSize_ = 1;
	}
	outNCHW_ = batchSize_*embedDim_*numPatches_;
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, embedDim_, patchRows_, patchCols_));
	checkCUDNN(cudnnSetTensor4dDescriptor(posDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, embedDim_, patchRows_, patchCols_));
}