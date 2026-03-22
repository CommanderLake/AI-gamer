#include "PatchEmbedLayer.h"
#include "NNCommon.h"
#include "CuCommon.cuh"
#include <vector>
#include <stdexcept>
PatchEmbedLayer::PatchEmbedLayer(cudnnHandle_t cudnnHandle, int batchSize, int inC, int inH, int inW, int patchSize, int embedDim, std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod, int framesPerSample, int channelsPerFrame) : cudnn_(cudnnHandle),
	batchSize_(batchSize), inC_(inC), inH_(inH), inW_(inW), patchSize_(patchSize), embedDim_(embedDim), framesPerSample_(framesPerSample), channelsPerFrame_(channelsPerFrame), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	if(framesPerSample_ <= 0) throw std::invalid_argument("framesPerSample must be positive");
	if(channelsPerFrame_ <= 0) throw std::invalid_argument("channelsPerFrame must be positive");
	if(inC_ != framesPerSample_*channelsPerFrame_) throw std::invalid_argument("PatchEmbedLayer temporal config must match input channels");
	layerName_ = layerName;
	train_ = train;
	patchRows_ = DivCeil(inH_, patchSize_);
	patchCols_ = DivCeil(inW_, patchSize_);
	patchArea_ = patchSize_*patchSize_;
	rawPatchDim_ = inC_*patchArea_;
	patchDim_ = channelsPerFrame_*patchArea_;
	numPatches_ = patchRows_*patchCols_;
	featureSize_ = embedDim_*numPatches_;
	outNCHW_ = batchSize_*featureSize_;
	alphaWeights_ = 1.0f/(batchSize_*numPatches_*gradAccumLength_);
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, batchSize_, embedDim_, patchRows_, patchCols_));
	checkCUDNN(cudnnCreateTensorDescriptor(&posDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(posDesc_, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 1, embedDim_, patchRows_, patchCols_));
	weightCount_ = embedDim_*patchDim_;
	posCount_ = featureSize_;
	offsetCount_ = offsetDim_*patchDim_;
	offsetEmbedCount_ = embedDim_*offsetDim_;
	temporalWeightCount_ = framesPerSample_*channelsPerFrame_;
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	CUDAMallocZero(&posEmbed_, posCount_*sizeof(__half));
	CUDAMallocZero(&offsetWeights_, offsetCount_*sizeof(__half));
	CUDAMallocZero(&offsetEmbedWeights_, offsetEmbedCount_*sizeof(__half));
	CUDAMallocZero(&temporalWeights_, temporalWeightCount_*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&patchBuffer_, batchSize_*numPatches_*rawPatchDim_*sizeof(__half));
	CUDAMallocZero(&fusedPatchBuffer_, batchSize_*numPatches_*patchDim_*sizeof(__half));
	if(train_) CUDAMallocZero(&patchGradBuffer_, batchSize_*numPatches_*rawPatchDim_*sizeof(__half));
	if(offsetDim_ > 0){
		CUDAMallocZero(&offsetActivations_, batchSize_*numPatches_*offsetDim_*sizeof(__half));
		if(train_) CUDAMallocZero(&offsetGrad_, batchSize_*numPatches_*offsetDim_*sizeof(__half));
	}
	std::vector<__half> temporalInit(temporalWeightCount_);
	for(int c = 0; c < channelsPerFrame_; ++c){
		for(int t = 0; t < framesPerSample_; ++t){ temporalInit[c*framesPerSample_ + t] = __float2half(1.0f/static_cast<float>(framesPerSample_)); }
	}
	checkCUDA(cudaMemcpy(temporalWeights_, temporalInit.data(), temporalWeightCount_*sizeof(__half), cudaMemcpyHostToDevice));
	if(train_){
		WeightInit(weights_, weightCount_, patchDim_, embedDim_, weightInitMethod);
		WeightInit(offsetWeights_, offsetCount_, patchDim_, offsetDim_, weightInitMethod);
		WeightInit(offsetEmbedWeights_, offsetEmbedCount_, offsetDim_, embedDim_, weightInitMethod);
		CUDAMallocZero(&gradWeights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&gradPosEmbed_, posCount_*sizeof(__half));
		CUDAMallocZero(&gradOffsetWeights_, offsetCount_*sizeof(__half));
		CUDAMallocZero(&gradOffsetEmbedWeights_, offsetEmbedCount_*sizeof(__half));
		CUDAMallocZero(&gradTemporalWeights_, temporalWeightCount_*sizeof(__half));
		CUDAMallocZero(&gradTemporalWeightsFloat_, temporalWeightCount_*sizeof(float));
		CUDAMallocZero(&outGrad_, batchSize_*inC_*inH_*inW_*sizeof(__half));
		CUDAMallocZero(&m_Weights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&v_Weights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&m_PosEmbed_, posCount_*sizeof(__half));
		CUDAMallocZero(&v_PosEmbed_, posCount_*sizeof(__half));
		CUDAMallocZero(&m_OffsetWeights_, offsetCount_*sizeof(__half));
		CUDAMallocZero(&v_OffsetWeights_, offsetCount_*sizeof(__half));
		CUDAMallocZero(&m_OffsetEmbed_, offsetEmbedCount_*sizeof(__half));
		CUDAMallocZero(&v_OffsetEmbed_, offsetEmbedCount_*sizeof(__half));
		CUDAMallocZero(&m_TemporalWeights_, temporalWeightCount_*sizeof(__half));
		CUDAMallocZero(&v_TemporalWeights_, temporalWeightCount_*sizeof(__half));
	}
}
PatchEmbedLayer::~PatchEmbedLayer(){
	cudaFree(weights_);
	cudaFree(posEmbed_);
	cudaFree(offsetWeights_);
	cudaFree(offsetEmbedWeights_);
	cudaFree(temporalWeights_);
	cudaFree(outData_);
	cudaFree(patchBuffer_);
	cudaFree(fusedPatchBuffer_);
	cudaFree(patchGradBuffer_);
	cudaFree(offsetActivations_);
	cudaFree(offsetGrad_);
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(posDesc_));
	if(train_){
		cudaFree(gradWeights_);
		cudaFree(gradPosEmbed_);
		cudaFree(gradOffsetWeights_);
		cudaFree(gradOffsetEmbedWeights_);
		cudaFree(gradTemporalWeights_);
		cudaFree(gradTemporalWeightsFloat_);
		cudaFree(outGrad_);
		cudaFree(m_Weights_);
		cudaFree(v_Weights_);
		cudaFree(m_PosEmbed_);
		cudaFree(v_PosEmbed_);
		cudaFree(m_OffsetWeights_);
		cudaFree(v_OffsetWeights_);
		cudaFree(m_OffsetEmbed_);
		cudaFree(v_OffsetEmbed_);
		cudaFree(m_TemporalWeights_);
		cudaFree(v_TemporalWeights_);
	}
}
__half* PatchEmbedLayer::Forward(__half* data){
	inData_ = data;
	ExtractPatches(data, patchBuffer_, batchSize_, inC_, inH_, inW_, patchSize_);
	TemporalFusePatches(patchBuffer_, fusedPatchBuffer_, batchSize_*numPatches_, framesPerSample_, channelsPerFrame_, patchArea_, temporalWeights_);
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_*numPatches_, patchDim_, &alpha_, weights_, CUDA_R_16F, embedDim_, fusedPatchBuffer_, CUDA_R_16F, patchDim_, &beta0_, outData_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	if(offsetDim_ > 0){
		checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, offsetDim_, batchSize_*numPatches_, patchDim_, &alpha_, offsetWeights_, CUDA_R_16F, offsetDim_, fusedPatchBuffer_, CUDA_R_16F, patchDim_, &beta0_, offsetActivations_, CUDA_R_16F, offsetDim_, CUDA_R_32F));
		TanhInPlace(offsetActivations_, batchSize_*numPatches_*offsetDim_);
		checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_*numPatches_, offsetDim_, &alpha_, offsetEmbedWeights_, CUDA_R_16F, embedDim_, offsetActivations_, CUDA_R_16F, offsetDim_, &beta1_, outData_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	}
	AddTensorBroadcast(alpha_, posEmbed_, alpha_, outData_, batchSize_, posCount_);
	return outData_;
}
__half* PatchEmbedLayer::Backward(__half* grad){
	const float* betaWeights = accumCount_++ % gradAccumLength_ == 0 ? &beta0_ : &beta1_;
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, patchDim_, batchSize_*numPatches_, &alphaWeights_, grad, CUDA_R_16F, embedDim_, fusedPatchBuffer_, CUDA_R_16F, patchDim_, betaWeights, gradWeights_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	if(offsetDim_ > 0){
		checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, offsetDim_, batchSize_*numPatches_, &alphaWeights_, grad, CUDA_R_16F, embedDim_, offsetActivations_, CUDA_R_16F, offsetDim_, betaWeights, gradOffsetEmbedWeights_, CUDA_R_16F, embedDim_, CUDA_R_32F));
		checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, offsetDim_, batchSize_*numPatches_, embedDim_, &alpha_, offsetEmbedWeights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &beta0_, offsetGrad_, CUDA_R_16F, offsetDim_, CUDA_R_32F));
		TanhBackward(offsetGrad_, offsetActivations_, batchSize_*numPatches_*offsetDim_);
		checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, offsetDim_, patchDim_, batchSize_*numPatches_, &alphaWeights_, offsetGrad_, CUDA_R_16F, offsetDim_, fusedPatchBuffer_, CUDA_R_16F, patchDim_, betaWeights, gradOffsetWeights_, CUDA_R_16F, offsetDim_, CUDA_R_32F));
	}
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, patchDim_, batchSize_*numPatches_, embedDim_, &alpha_, weights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &beta0_, fusedPatchBuffer_, CUDA_R_16F, patchDim_, CUDA_R_32F));
	if(offsetDim_ > 0){
		checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, patchDim_, batchSize_*numPatches_, offsetDim_, &alpha_, offsetWeights_, CUDA_R_16F, offsetDim_, offsetGrad_, CUDA_R_16F, offsetDim_, &beta1_, fusedPatchBuffer_, CUDA_R_16F, patchDim_, CUDA_R_32F));
	}
	const bool zeroPos = ((accumCount_ - 1) % gradAccumLength_) == 0;
	TemporalWeightGrad(patchBuffer_, fusedPatchBuffer_, gradTemporalWeightsFloat_, batchSize_*numPatches_, framesPerSample_, channelsPerFrame_, patchArea_, zeroPos, alphaWeights_);
	ConvertFloatToHalf(gradTemporalWeightsFloat_, gradTemporalWeights_, temporalWeightCount_);
	TemporalUnfusePatchGrads(fusedPatchBuffer_, patchGradBuffer_, batchSize_*numPatches_, framesPerSample_, channelsPerFrame_, patchArea_, temporalWeights_);
	SumPositionalGrad(grad, gradPosEmbed_, batchSize_, embedDim_, numPatches_, zeroPos, alphaWeights_);
	CombinePatchGrads(patchGradBuffer_, outGrad_, batchSize_, inC_, inH_, inW_, patchSize_);
	return outGrad_;
}
void PatchEmbedLayer::UpdateParameters(float lr){
	if(accumCount_ % gradAccumLength_ > 0) return;
	if(useAdamW_){
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, lr, t_, weightDecay_, weightCount_);
		AdamWHalf(posEmbed_, gradPosEmbed_, m_PosEmbed_, v_PosEmbed_, lr, t_, 0.0f, posCount_);
		AdamWHalf(temporalWeights_, gradTemporalWeights_, m_TemporalWeights_, v_TemporalWeights_, lr, t_, 0.0f, temporalWeightCount_);
		if(offsetDim_ > 0){
			AdamWHalf(offsetWeights_, gradOffsetWeights_, m_OffsetWeights_, v_OffsetWeights_, lr, t_, weightDecay_, offsetCount_);
			AdamWHalf(offsetEmbedWeights_, gradOffsetEmbedWeights_, m_OffsetEmbed_, v_OffsetEmbed_, lr, t_, 0.0f, offsetEmbedCount_);
		}
	} else{
		SGDHalf(weights_, gradWeights_, weightCount_, lr, weightDecay_);
		SGDHalf(posEmbed_, gradPosEmbed_, posCount_, lr, 0.0f);
		SGDHalf(temporalWeights_, gradTemporalWeights_, temporalWeightCount_, lr, 0.0f);
		if(offsetDim_ > 0){
			SGDHalf(offsetWeights_, gradOffsetWeights_, offsetCount_, lr, weightDecay_);
			SGDHalf(offsetEmbedWeights_, gradOffsetEmbedWeights_, offsetEmbedCount_, lr, 0.0f);
		}
	}
	++t_;
}
void PatchEmbedLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, posEmbed_, posCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), posCount_*sizeof(__half));
	cudaMemcpy(buffer, offsetWeights_, offsetCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), offsetCount_*sizeof(__half));
	cudaMemcpy(buffer, offsetEmbedWeights_, offsetEmbedCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), offsetEmbedCount_*sizeof(__half));
	cudaMemcpy(buffer, temporalWeights_, temporalWeightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), temporalWeightCount_*sizeof(__half));
}
void PatchEmbedLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), posCount_*sizeof(__half));
	cudaMemcpy(posEmbed_, buffer, posCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), offsetCount_*sizeof(__half));
	cudaMemcpy(offsetWeights_, buffer, offsetCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), offsetEmbedCount_*sizeof(__half));
	cudaMemcpy(offsetEmbedWeights_, buffer, offsetEmbedCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), temporalWeightCount_*sizeof(__half));
	cudaMemcpy(temporalWeights_, buffer, temporalWeightCount_*sizeof(__half), cudaMemcpyHostToDevice);
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
	cudaMemcpy(buffer, m_TemporalWeights_, temporalWeightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), temporalWeightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_TemporalWeights_, temporalWeightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), temporalWeightCount_*sizeof(__half));
	if(offsetDim_ > 0){
		cudaMemcpy(buffer, m_OffsetWeights_, offsetCount_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), offsetCount_*sizeof(__half));
		cudaMemcpy(buffer, v_OffsetWeights_, offsetCount_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), offsetCount_*sizeof(__half));
		cudaMemcpy(buffer, m_OffsetEmbed_, offsetEmbedCount_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), offsetEmbedCount_*sizeof(__half));
		cudaMemcpy(buffer, v_OffsetEmbed_, offsetEmbedCount_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), offsetEmbedCount_*sizeof(__half));
	}
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
	file.read(reinterpret_cast<char*>(buffer), temporalWeightCount_*sizeof(__half));
	cudaMemcpy(m_TemporalWeights_, buffer, temporalWeightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), temporalWeightCount_*sizeof(__half));
	cudaMemcpy(v_TemporalWeights_, buffer, temporalWeightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	if(offsetDim_ > 0){
		file.read(reinterpret_cast<char*>(buffer), offsetCount_*sizeof(__half));
		cudaMemcpy(m_OffsetWeights_, buffer, offsetCount_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), offsetCount_*sizeof(__half));
		cudaMemcpy(v_OffsetWeights_, buffer, offsetCount_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), offsetEmbedCount_*sizeof(__half));
		cudaMemcpy(m_OffsetEmbed_, buffer, offsetEmbedCount_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), offsetEmbedCount_*sizeof(__half));
		cudaMemcpy(v_OffsetEmbed_, buffer, offsetEmbedCount_*sizeof(__half), cudaMemcpyHostToDevice);
	}
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t PatchEmbedLayer::GetParameterSize(){ return (weightCount_ + posCount_ + offsetCount_ + offsetEmbedCount_ + temporalWeightCount_)*sizeof(__half); }
size_t PatchEmbedLayer::GetOptimizerStateSize(){ return useAdamW_ ? (weightCount_ + posCount_ + offsetCount_ + offsetEmbedCount_ + temporalWeightCount_)*sizeof(__half)*2 + sizeof(int) : 0; }
void PatchEmbedLayer::SetTrain(const bool enable){ train_ = enable; }

void PatchEmbedLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!useAdamW_ || !train_) return;
	halfTasks.push_back({weights_, gradWeights_, m_Weights_, v_Weights_, static_cast<int>(weightCount_), weightDecay_});
	halfTasks.push_back({posEmbed_, gradPosEmbed_, m_PosEmbed_, v_PosEmbed_, posCount_, 0.0f});
	halfTasks.push_back({temporalWeights_, gradTemporalWeights_, m_TemporalWeights_, v_TemporalWeights_, temporalWeightCount_, 0.0f});
	if(offsetCount_ > 0){
		halfTasks.push_back({offsetWeights_, gradOffsetWeights_, m_OffsetWeights_, v_OffsetWeights_, offsetCount_, weightDecay_});
		halfTasks.push_back({offsetEmbedWeights_, gradOffsetEmbedWeights_, m_OffsetEmbed_, v_OffsetEmbed_, offsetEmbedCount_, 0.0f});
	}
}
