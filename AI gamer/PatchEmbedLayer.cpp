#include "PatchEmbedLayer.h"
#include "common.h"
#include "CuCommon.cuh"
PatchEmbedLayer::PatchEmbedLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inC, int inH, int inW, int patchSize, int controlTokens, int controlInputDim, int embedDim, const char* layerName, bool train, float weightDecay, int gradAccumLength,
								WeightInitMethod weightInitMethod) : cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(batchSize), inC_(inC), inH_(inH), inW_(inW), patchSize_(patchSize), embedDim_(embedDim), controlTokens_(controlTokens), controlInputDim_(controlInputDim),
																	weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	patchRows_ = DivCeil(inH_, patchSize_);
	patchCols_ = DivCeil(inW_, patchSize_);
	patchDim_ = inC_ * patchSize_ * patchSize_;
	numPatches_ = patchRows_ * patchCols_;
	featureSize_ = embedDim_ * numPatches_;
	totalTokens_ = numPatches_ + controlTokens_;
	outNCHW_ = batchSize_ * embedDim_ * totalTokens_;
	batchControlCount_ = batchSize_ * controlTokens_;
	controlInputCount_ = static_cast<size_t>(batchControlCount_) * controlInputDim_;
	controlOutputCount_ = static_cast<size_t>(batchControlCount_) * embedDim_;
	alphaWeights_ = 1.0f / (batchSize_ * gradAccumLength_);
	controlAlphaWeights_ = (batchControlCount_ > 0) ? 1.0f / (static_cast<float>(batchControlCount_) * gradAccumLength_) : 0.0f;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, batchSize_, embedDim_, patchRows_, patchCols_));
	checkCUDNN(cudnnCreateTensorDescriptor(&posDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(posDesc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, embedDim_, patchRows_, patchCols_));
	weightCount_ = embedDim_ * patchDim_;
	posCount_ = featureSize_;
	controlPosCount_ = controlTokens_ > 0 ? static_cast<size_t>(controlTokens_) * embedDim_ : 0;
	tokenTypeClasses_ = controlTokens_ > 0 ? 2 : 1;
	tokenTypeCount_ = static_cast<size_t>(tokenTypeClasses_) * embedDim_;
	CUDAMallocZero(&weights_, weightCount_ * sizeof(__half));
	CUDAMallocZero(&posEmbed_, posCount_ * sizeof(__half));
	if(controlPosCount_ > 0){ CUDAMallocZero(&controlPosEmbed_, controlPosCount_ * sizeof(__half)); }
	if(tokenTypeCount_ > 0){ CUDAMallocZero(&tokenTypeEmbed_, tokenTypeCount_ * sizeof(__half)); }
	CUDAMallocZero(&outData_, outNCHW_ * sizeof(__half));
	CUDAMallocZero(&patchBuffer_, batchSize_ * numPatches_ * patchDim_ * sizeof(__half));
	CUDAMallocZero(&patchTokens_, batchSize_ * featureSize_ * sizeof(__half));
	if(controlInputCount_ > 0){ CUDAMallocZero(&controlWeights_, static_cast<size_t>(controlInputDim_) * embedDim_ * sizeof(__half)); }
	if(controlOutputCount_ > 0){ CUDAMallocZero(&controlBiases_, embedDim_ * sizeof(__half)); }
	if(train_){
		WeightInit(weights_, weightCount_, patchDim_, embedDim_, weightInitMethod);
		if(controlWeights_){ WeightInit(controlWeights_, static_cast<size_t>(controlInputDim_) * embedDim_, controlInputDim_, embedDim_, weightInitMethod); }
		CUDAMallocZero(&gradWeights_, weightCount_ * sizeof(__half));
		CUDAMallocZero(&gradPosEmbed_, posCount_ * sizeof(__half));
		if(controlPosCount_ > 0){ CUDAMallocZero(&gradControlPosEmbed_, controlPosCount_ * sizeof(__half)); }
		if(tokenTypeCount_ > 0){ CUDAMallocZero(&gradTokenTypeEmbed_, tokenTypeCount_ * sizeof(__half)); }
		CUDAMallocZero(&outGrad_, batchSize_ * inC_ * inH_ * inW_ * sizeof(__half));
		CUDAMallocZero(&patchGradBuffer_, batchSize_ * featureSize_ * sizeof(__half));
		CUDAMallocZero(&m_Weights_, weightCount_ * sizeof(__half));
		CUDAMallocZero(&v_Weights_, weightCount_ * sizeof(__half));
		CUDAMallocZero(&m_PosEmbed_, posCount_ * sizeof(__half));
		CUDAMallocZero(&v_PosEmbed_, posCount_ * sizeof(__half));
		if(controlPosCount_ > 0){
			CUDAMallocZero(&m_ControlPosEmbed_, controlPosCount_ * sizeof(__half));
			CUDAMallocZero(&v_ControlPosEmbed_, controlPosCount_ * sizeof(__half));
		}
		if(tokenTypeCount_ > 0){
			CUDAMallocZero(&m_TokenTypeEmbed_, tokenTypeCount_ * sizeof(__half));
			CUDAMallocZero(&v_TokenTypeEmbed_, tokenTypeCount_ * sizeof(__half));
		}
		if(controlWeights_){
			CUDAMallocZero(&gradControlWeights_, static_cast<size_t>(controlInputDim_) * embedDim_ * sizeof(__half));
			CUDAMallocZero(&m_ControlWeights_, static_cast<size_t>(controlInputDim_) * embedDim_ * sizeof(__half));
			CUDAMallocZero(&v_ControlWeights_, static_cast<size_t>(controlInputDim_) * embedDim_ * sizeof(__half));
			if(controlBiases_){
				CUDAMallocZero(&gradControlBiases_, embedDim_ * sizeof(__half));
				CUDAMallocZero(&m_ControlBiases_, embedDim_ * sizeof(__half));
				CUDAMallocZero(&v_ControlBiases_, embedDim_ * sizeof(__half));
			}
		}
	}
}
PatchEmbedLayer::~PatchEmbedLayer(){
	cudaFree(weights_);
	cudaFree(posEmbed_);
	cudaFree(outData_);
	cudaFree(patchBuffer_);
	cudaFree(patchTokens_);
	if(controlPosEmbed_){ cudaFree(controlPosEmbed_); }
	if(tokenTypeEmbed_){ cudaFree(tokenTypeEmbed_); }
	if(controlWeights_){ cudaFree(controlWeights_); }
	if(controlBiases_){ cudaFree(controlBiases_); }
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	if(train_){
		cudaFree(gradWeights_);
		cudaFree(gradPosEmbed_);
		if(gradControlPosEmbed_){ cudaFree(gradControlPosEmbed_); }
		if(gradTokenTypeEmbed_){ cudaFree(gradTokenTypeEmbed_); }
		cudaFree(outGrad_);
		cudaFree(patchGradBuffer_);
		cudaFree(m_Weights_);
		cudaFree(v_Weights_);
		cudaFree(m_PosEmbed_);
		cudaFree(v_PosEmbed_);
		if(m_ControlPosEmbed_){ cudaFree(m_ControlPosEmbed_); }
		if(v_ControlPosEmbed_){ cudaFree(v_ControlPosEmbed_); }
		if(m_TokenTypeEmbed_){ cudaFree(m_TokenTypeEmbed_); }
		if(v_TokenTypeEmbed_){ cudaFree(v_TokenTypeEmbed_); }
		if(controlWeights_){
			cudaFree(gradControlWeights_);
			cudaFree(m_ControlWeights_);
			cudaFree(v_ControlWeights_);
		}
		if(controlBiases_){
			cudaFree(gradControlBiases_);
			cudaFree(m_ControlBiases_);
			cudaFree(v_ControlBiases_);
		}
	}
}
__half* PatchEmbedLayer::Forward(__half* data){
	inData_ = data;
	controlInput_ = nullptr;
	const size_t imageCount = static_cast<size_t>(batchSize_) * inC_ * inH_ * inW_;
	if(controlInputCount_ > 0){ controlInput_ = data + imageCount; }
	ExtractPatches(data, patchBuffer_, batchSize_, inC_, inH_, inW_, patchSize_);
	checkCUBLAS(cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, batchSize_*numPatches_, patchDim_, &alpha_, weights_, CUDA_R_16F, embedDim_, patchBuffer_, CUDA_R_16F, patchDim_, &beta0_, outData_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUDNN(cudnnAddTensor(cudnn_, &alpha_, posDesc_, posEmbed_, &alpha_, outDesc_, outData_));
	if(tokenTypeEmbed_ && batchSize_ > 0 && numPatches_ > 0){ AddBias(outData_, tokenTypeEmbed_, embedDim_, batchSize_ * numPatches_); }
	if(controlInputCount_ > 0){
		auto* controlOut = outData_ + static_cast<size_t>(batchSize_) * featureSize_;
		if(controlWeights_ && controlInput_){
			checkCUBLAS(
				cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, embedDim_, batchControlCount_, controlInputDim_, &alpha_, controlWeights_, CUDA_R_16F, embedDim_, controlInput_, CUDA_R_16F, controlInputDim_, &beta0_, controlOut, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
			if(controlBiases_){ AddBias(controlOut, controlBiases_, embedDim_, batchControlCount_); }
		} else{ checkCUDA(cudaMemset(controlOut, 0, controlOutputCount_*sizeof(__half))); }
		if(controlPosEmbed_){ AddPerTokenEmbedding(controlOut, controlPosEmbed_, batchSize_, controlTokens_, embedDim_); }
		if(tokenTypeEmbed_ && tokenTypeClasses_ > 1){ AddBias(controlOut, tokenTypeEmbed_ + embedDim_, embedDim_, batchControlCount_); }
	}
	return outData_;
}
__half* PatchEmbedLayer::Backward(__half* grad){
	const bool resetAccum = (accumCount_ % gradAccumLength_) == 0;
	const float* betaWeights = resetAccum ? &beta0_ : &beta1_;
	++accumCount_;
	const auto patchCount = batchSize_ * numPatches_;
	const __half* gradControl = nullptr;
	if(controlTokens_ > 0){ gradControl = grad + static_cast<size_t>(batchSize_) * featureSize_; }
	checkCUBLAS(cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, patchDim_, patchCount, &alphaWeights_, grad, CUDA_R_16F, embedDim_, patchBuffer_, CUDA_R_16F, patchDim_, betaWeights, gradWeights_, CUDA_R_16F, embedDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, patchDim_, patchCount, embedDim_, &alpha_, weights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &beta0_, patchBuffer_, CUDA_R_16F, patchDim_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	SumPositionalGrad(grad, gradPosEmbed_, batchSize_, embedDim_, numPatches_, resetAccum, alphaWeights_);
	if(controlPosEmbed_ && gradControlPosEmbed_ && gradControl){ SumPositionalGrad(gradControl, gradControlPosEmbed_, batchSize_, embedDim_, controlTokens_, resetAccum, controlAlphaWeights_); }
	if(gradTokenTypeEmbed_){
		AccumulateBiasGrad(grad, gradTokenTypeEmbed_, embedDim_, batchSize_ * numPatches_, alphaWeights_, resetAccum);
		if(gradControl && tokenTypeClasses_ > 1){ AccumulateBiasGrad(gradControl, gradTokenTypeEmbed_ + embedDim_, embedDim_, batchControlCount_, controlAlphaWeights_, resetAccum); }
	}
	if(controlInputCount_ > 0 && controlWeights_ && controlInput_ && gradControlWeights_ && gradControl){
		checkCUBLAS(
			cublasGemmEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_T, embedDim_, controlInputDim_, batchControlCount_, &controlAlphaWeights_, gradControl, CUDA_R_16F, embedDim_, controlInput_, CUDA_R_16F, controlInputDim_, betaWeights, gradControlWeights_, CUDA_R_16F, embedDim_, CUDA_R_32F,
				CUBLAS_GEMM_DEFAULT_TENSOR_OP));
		if(controlBiases_ && gradControlBiases_){ AccumulateBiasGrad(gradControl, gradControlBiases_, embedDim_, batchControlCount_, controlAlphaWeights_, resetAccum); }
	}
	CombinePatchGrads(patchBuffer_, outGrad_, batchSize_, inC_, inH_, inW_, patchSize_);
	return outGrad_;
}
void PatchEmbedLayer::UpdateParameters(float lr){
	if(accumCount_ % gradAccumLength_ > 0) return;
	if(useAdamW_){
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, lr, t_, weightDecay_, weightCount_);
		AdamWHalf(posEmbed_, gradPosEmbed_, m_PosEmbed_, v_PosEmbed_, lr, t_, 0.0f, posCount_);
		if(controlPosEmbed_ && gradControlPosEmbed_){ AdamWHalf(controlPosEmbed_, gradControlPosEmbed_, m_ControlPosEmbed_, v_ControlPosEmbed_, lr, t_, 0.0f, static_cast<int>(controlPosCount_)); }
		if(tokenTypeEmbed_ && gradTokenTypeEmbed_){ AdamWHalf(tokenTypeEmbed_, gradTokenTypeEmbed_, m_TokenTypeEmbed_, v_TokenTypeEmbed_, lr, t_, 0.0f, static_cast<int>(tokenTypeCount_)); }
		if(controlWeights_){ AdamWHalf(controlWeights_, gradControlWeights_, m_ControlWeights_, v_ControlWeights_, lr, t_, weightDecay_, static_cast<size_t>(controlInputDim_) * embedDim_); }
		if(controlBiases_){ AdamWHalf(controlBiases_, gradControlBiases_, m_ControlBiases_, v_ControlBiases_, lr, t_, 0.0f, embedDim_); }
	} else{
		SGDHalf(weights_, gradWeights_, weightCount_, lr, weightDecay_);
		SGDHalf(posEmbed_, gradPosEmbed_, posCount_, lr, 0.0f);
		if(controlPosEmbed_ && gradControlPosEmbed_){ SGDHalf(controlPosEmbed_, gradControlPosEmbed_, static_cast<int>(controlPosCount_), lr, 0.0f); }
		if(tokenTypeEmbed_ && gradTokenTypeEmbed_){ SGDHalf(tokenTypeEmbed_, gradTokenTypeEmbed_, static_cast<int>(tokenTypeCount_), lr, 0.0f); }
		if(controlWeights_){ SGDHalf(controlWeights_, gradControlWeights_, static_cast<size_t>(controlInputDim_) * embedDim_, lr, weightDecay_); }
		if(controlBiases_){ SGDHalf(controlBiases_, gradControlBiases_, embedDim_, lr, 0.0f); }
	}
	++t_;
}
void PatchEmbedLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(buffer, posEmbed_, posCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), posCount_ * sizeof(__half));
	if(controlPosEmbed_ && controlPosCount_ > 0){
		cudaMemcpy(buffer, controlPosEmbed_, controlPosCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), controlPosCount_ * sizeof(__half));
	}
	if(tokenTypeEmbed_ && tokenTypeCount_ > 0){
		cudaMemcpy(buffer, tokenTypeEmbed_, tokenTypeCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), tokenTypeCount_ * sizeof(__half));
	}
	if(controlWeights_){
		const size_t controlWeightCount = static_cast<size_t>(controlInputDim_) * embedDim_;
		cudaMemcpy(buffer, controlWeights_, controlWeightCount * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), controlWeightCount * sizeof(__half));
	}
	if(controlBiases_){
		cudaMemcpy(buffer, controlBiases_, embedDim_ * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), embedDim_ * sizeof(__half));
	}
}
void PatchEmbedLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), posCount_ * sizeof(__half));
	cudaMemcpy(posEmbed_, buffer, posCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	if(controlPosEmbed_ && controlPosCount_ > 0){
		file.read(reinterpret_cast<char*>(buffer), controlPosCount_ * sizeof(__half));
		cudaMemcpy(controlPosEmbed_, buffer, controlPosCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	}
	if(tokenTypeEmbed_ && tokenTypeCount_ > 0){
		file.read(reinterpret_cast<char*>(buffer), tokenTypeCount_ * sizeof(__half));
		cudaMemcpy(tokenTypeEmbed_, buffer, tokenTypeCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	}
	if(controlWeights_){
		const size_t controlWeightCount = static_cast<size_t>(controlInputDim_) * embedDim_;
		file.read(reinterpret_cast<char*>(buffer), controlWeightCount * sizeof(__half));
		cudaMemcpy(controlWeights_, buffer, controlWeightCount * sizeof(__half), cudaMemcpyHostToDevice);
	}
	if(controlBiases_){
		file.read(reinterpret_cast<char*>(buffer), embedDim_ * sizeof(__half));
		cudaMemcpy(controlBiases_, buffer, embedDim_ * sizeof(__half), cudaMemcpyHostToDevice);
	}
}
void PatchEmbedLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(buffer, m_PosEmbed_, posCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), posCount_ * sizeof(__half));
	cudaMemcpy(buffer, v_PosEmbed_, posCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), posCount_ * sizeof(__half));
	if(controlPosEmbed_ && m_ControlPosEmbed_ && v_ControlPosEmbed_){
		cudaMemcpy(buffer, m_ControlPosEmbed_, controlPosCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), controlPosCount_ * sizeof(__half));
		cudaMemcpy(buffer, v_ControlPosEmbed_, controlPosCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), controlPosCount_ * sizeof(__half));
	}
	if(tokenTypeEmbed_ && m_TokenTypeEmbed_ && v_TokenTypeEmbed_){
		cudaMemcpy(buffer, m_TokenTypeEmbed_, tokenTypeCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), tokenTypeCount_ * sizeof(__half));
		cudaMemcpy(buffer, v_TokenTypeEmbed_, tokenTypeCount_ * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), tokenTypeCount_ * sizeof(__half));
	}
	if(controlWeights_ && m_ControlWeights_ && v_ControlWeights_){
		const size_t controlWeightCount = static_cast<size_t>(controlInputDim_) * embedDim_;
		cudaMemcpy(buffer, m_ControlWeights_, controlWeightCount * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), controlWeightCount * sizeof(__half));
		cudaMemcpy(buffer, v_ControlWeights_, controlWeightCount * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), controlWeightCount * sizeof(__half));
	}
	if(controlBiases_ && m_ControlBiases_ && v_ControlBiases_){
		cudaMemcpy(buffer, m_ControlBiases_, embedDim_ * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), embedDim_ * sizeof(__half));
		cudaMemcpy(buffer, v_ControlBiases_, embedDim_ * sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), embedDim_ * sizeof(__half));
	}
	file.write(reinterpret_cast<char*>(&t_), sizeof(int));
}
void PatchEmbedLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_ * sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), posCount_ * sizeof(__half));
	cudaMemcpy(m_PosEmbed_, buffer, posCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), posCount_ * sizeof(__half));
	cudaMemcpy(v_PosEmbed_, buffer, posCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	if(controlPosEmbed_ && m_ControlPosEmbed_ && v_ControlPosEmbed_){
		file.read(reinterpret_cast<char*>(buffer), controlPosCount_ * sizeof(__half));
		cudaMemcpy(m_ControlPosEmbed_, buffer, controlPosCount_ * sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), controlPosCount_ * sizeof(__half));
		cudaMemcpy(v_ControlPosEmbed_, buffer, controlPosCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	}
	if(tokenTypeEmbed_ && m_TokenTypeEmbed_ && v_TokenTypeEmbed_){
		file.read(reinterpret_cast<char*>(buffer), tokenTypeCount_ * sizeof(__half));
		cudaMemcpy(m_TokenTypeEmbed_, buffer, tokenTypeCount_ * sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), tokenTypeCount_ * sizeof(__half));
		cudaMemcpy(v_TokenTypeEmbed_, buffer, tokenTypeCount_ * sizeof(__half), cudaMemcpyHostToDevice);
	}
	if(controlWeights_ && m_ControlWeights_ && v_ControlWeights_){
		const size_t controlWeightCount = static_cast<size_t>(controlInputDim_) * embedDim_;
		file.read(reinterpret_cast<char*>(buffer), controlWeightCount * sizeof(__half));
		cudaMemcpy(m_ControlWeights_, buffer, controlWeightCount * sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), controlWeightCount * sizeof(__half));
		cudaMemcpy(v_ControlWeights_, buffer, controlWeightCount * sizeof(__half), cudaMemcpyHostToDevice);
	}
	if(controlBiases_ && m_ControlBiases_ && v_ControlBiases_){
		file.read(reinterpret_cast<char*>(buffer), embedDim_ * sizeof(__half));
		cudaMemcpy(m_ControlBiases_, buffer, embedDim_ * sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), embedDim_ * sizeof(__half));
		cudaMemcpy(v_ControlBiases_, buffer, embedDim_ * sizeof(__half), cudaMemcpyHostToDevice);
	}
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t PatchEmbedLayer::GetParameterSize(){
	const size_t controlWeightCount = controlWeights_ ? static_cast<size_t>(controlInputDim_) * embedDim_ : 0;
	const size_t controlBiasCount = controlBiases_ ? static_cast<size_t>(embedDim_) : 0;
	return (weightCount_ + posCount_ + controlPosCount_ + tokenTypeCount_ + controlWeightCount + controlBiasCount) * sizeof(__half);
}
size_t PatchEmbedLayer::GetOptimizerStateSize(){
	if(!useAdamW_) return 0;
	const size_t controlWeightCount = controlWeights_ ? static_cast<size_t>(controlInputDim_) * embedDim_ : 0;
	const size_t controlBiasCount = controlBiases_ ? static_cast<size_t>(embedDim_) : 0;
	return (weightCount_ + posCount_ + controlPosCount_ + tokenTypeCount_ + controlWeightCount + controlBiasCount) * sizeof(__half) * 2 + sizeof(int);
}
void PatchEmbedLayer::SetTrain(const bool enable){ train_ = enable; }