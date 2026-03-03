#include "WmmaAttentionLayer.h"
#include "NNCommon.h"
#include "CuCommon.cuh"
#include <stdexcept>
WmmaAttentionLayer::WmmaAttentionLayer(cudnnHandle_t cudnnHandle, int batchSize, int tokens, int embedDim, int numHeads, std::string layerName, bool train, float weightDecay, const int gradAccumLength, WeightInitMethod weightInitMethod) :
	WmmaAttentionLayer(cudnnHandle, batchSize, tokens, embedDim, numHeads, std::move(layerName), train, weightDecay, gradAccumLength, weightInitMethod, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr){}
WmmaAttentionLayer::WmmaAttentionLayer(cudnnHandle_t cudnnHandle, int batchSize, int tokens, int embedDim, int numHeads, std::string layerName, bool train, float weightDecay, const int gradAccumLength, WeightInitMethod weightInitMethod,
	__half* sharedWorkspace, __half* sharedQPacked, __half* sharedKPacked, __half* sharedVPacked, __half* sharedAttnOutPacked, __half* sharedDQPacked, __half* sharedDKPacked, __half* sharedDVPacked,
	float* sharedAttnGradWorkspace) : cudnnHandle_(cudnnHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), numHeads_(numHeads), gradAccumLength_(gradAccumLength), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	headDim_ = embedDim_/numHeads_;
	outNCHW_ = batchSize_*tokens_*embedDim_;
	alphaWeights_ = 1.0f/(batchSize_*tokens_*gradAccumLength_);
	const size_t projSize = embedDim_*embedDim_;
	CUDAMallocZero(&qkvWeightsBase_, 3*projSize*sizeof(__half));
	qWeights_ = qkvWeightsBase_;
	kWeights_ = qkvWeightsBase_ + projSize;
	vWeights_ = qkvWeightsBase_ + 2*projSize;
	CUDAMallocZero(&oWeights_, projSize*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	const auto attentionElems = static_cast<size_t>(batchSize_)*tokens_*tokens_*numHeads_;
	workspace_ = sharedWorkspace;
	qPacked_ = sharedQPacked;
	kPacked_ = sharedKPacked;
	vPacked_ = sharedVPacked;
	attnOutPacked_ = sharedAttnOutPacked;
	if(workspace_ == nullptr || qPacked_ == nullptr || kPacked_ == nullptr || vPacked_ == nullptr || attnOutPacked_ == nullptr){
		ownsTemporaries_ = true;
		CUDAMallocZero(&workspace_, 4*outNCHW_*sizeof(__half) + attentionElems*sizeof(__half));
		CUDAMallocZero(&qPacked_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&kPacked_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&vPacked_, outNCHW_*sizeof(__half));
		CUDAMallocZero(&attnOutPacked_, outNCHW_*sizeof(__half));
	} else{ ownsTemporaries_ = false; }
	if(train_){
		WeightInit(qWeights_, projSize, embedDim_, embedDim_, weightInitMethod);
		WeightInit(kWeights_, projSize, embedDim_, embedDim_, weightInitMethod);
		WeightInit(vWeights_, projSize, embedDim_, embedDim_, weightInitMethod);
		WeightInit(oWeights_, projSize, embedDim_, embedDim_, weightInitMethod);
		const auto gradWorkspaceElems = static_cast<size_t>(batchSize_)*tokens_*tokens_*numHeads_;
		attnGradWorkspaceSize_ = gradWorkspaceElems;
		attnGradWorkspace_ = sharedAttnGradWorkspace;
		if(gradWorkspaceElems > 0 && attnGradWorkspace_ == nullptr){
			ownsAttnGradWorkspace_ = true;
			CUDAMallocZero(&attnGradWorkspace_, gradWorkspaceElems*sizeof(float));
		} else{ ownsAttnGradWorkspace_ = false; }
		CUDAMallocZero(&gradQkvBase_, 3*projSize*sizeof(__half));
		gradQ_ = gradQkvBase_;
		gradK_ = gradQkvBase_ + projSize;
		gradV_ = gradQkvBase_ + 2*projSize;
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
		dQPacked_ = sharedDQPacked;
		dKPacked_ = sharedDKPacked;
		dVPacked_ = sharedDVPacked;
		if(dQPacked_ == nullptr || dKPacked_ == nullptr || dVPacked_ == nullptr){
			ownsPackedGradTemporaries_ = true;
			CUDAMallocZero(&dQPacked_, outNCHW_*sizeof(__half));
			CUDAMallocZero(&dKPacked_, outNCHW_*sizeof(__half));
			CUDAMallocZero(&dVPacked_, outNCHW_*sizeof(__half));
		} else{ ownsPackedGradTemporaries_ = false; }
	}
}
WmmaAttentionLayer::~WmmaAttentionLayer(){
	cudaFree(qkvWeightsBase_);
	cudaFree(oWeights_);
	cudaFree(outData_);
	if(ownsTemporaries_){
		cudaFree(workspace_);
		cudaFree(qPacked_);
		cudaFree(kPacked_);
		cudaFree(vPacked_);
		cudaFree(attnOutPacked_);
	}
	cudaFree(relPosBias_);
	cudaFree(relPosIndex_);
	if(train_){
		if(ownsAttnGradWorkspace_){ cudaFree(attnGradWorkspace_); }
		cudaFree(gradQkvBase_);
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
		if(ownsPackedGradTemporaries_){
			cudaFree(dQPacked_);
			cudaFree(dKPacked_);
			cudaFree(dVPacked_);
		}
		cudaFree(gradRelPosBias_);
		cudaFree(m_relPosBias_);
		cudaFree(v_relPosBias_);
	}
}
__half* WmmaAttentionLayer::Forward(__half* data){
	const auto Q = workspace_;
	const auto K = workspace_ + outNCHW_;
	const auto V = workspace_ + 2*outNCHW_;
	const auto attnOut = workspace_ + 3*outNCHW_;
	const auto attentionWeights = workspace_ + 4*outNCHW_;
	inData_ = data;
	const long long qkvStrideA = static_cast<long long>(embedDim_)*embedDim_;
	const long long qkvStrideC = static_cast<long long>(outNCHW_);
	checkCLNN(CLNNGemmStridedBatchedEx(CLNN_OP_N, CLNN_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, qWeights_, CUDA_R_16F, embedDim_, qkvStrideA, data, CUDA_R_16F, embedDim_, 0, &zero_, Q, CUDA_R_16F, embedDim_, qkvStrideC, 3, CUDA_R_32F));
	PackColumnsToHeads(Q, K, V, qPacked_, kPacked_, vPacked_, batchSize_, tokens_, embedDim_, numHeads_);
	dQ = Q;
	dK = K;
	dV = V;
	const float* relPosBias = useRelPosBias_ ? relPosBias_ : nullptr;
	const int* relPosIndex = useRelPosBias_ ? relPosIndex_ : nullptr;
	WmmaAttention(qPacked_, kPacked_, vPacked_, attnOutPacked_, train_ ? attentionWeights : nullptr, attentionMask_, relPosBias, relPosIndex, relPosSize_, batchSize_, tokens_, headDim_, numHeads_, maskBatchSize_, maskHeads_);
	PackHeadsToColumns(attnOutPacked_, attnOut, batchSize_, tokens_, embedDim_, numHeads_);
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, oWeights_, CUDA_R_16F, embedDim_, attnOut, CUDA_R_16F, embedDim_, &zero_, outData_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	return outData_;
}
__half* WmmaAttentionLayer::Backward(__half* grad){
	const auto attnOut = workspace_ + 3*outNCHW_;
	const __half* attentionWeights = workspace_ + 4*outNCHW_;
	const float* betaWeights = accumCount_++ % gradAccumLength_ == 0 ? &zero_ : &one_;
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, grad, CUDA_R_16F, embedDim_, attnOut, CUDA_R_16F, embedDim_, betaWeights, gradOut_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, oWeights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &zero_, attnOut, CUDA_R_16F, embedDim_, CUDA_R_32F));
	PackColumnsToHeads(attnOut, attnOutPacked_, batchSize_, tokens_, embedDim_, numHeads_);
	WmmaAttentionBackward(qPacked_, kPacked_, vPacked_, attnOutPacked_, attentionWeights, dQPacked_, dKPacked_, dVPacked_, attnGradWorkspace_, attnGradWorkspaceSize_, batchSize_, tokens_, headDim_, numHeads_);
	if(useRelPosBias_ && train_ && gradRelPosBias_ && relPosIndex_){
		const bool resetGrad = (accumCount_ - 1) % gradAccumLength_ == 0;
		if(resetGrad){
			const size_t gradSizeBytes = static_cast<size_t>(numHeads_) * relPosSize_ * sizeof(float);
			checkCUDA(cudaMemset(gradRelPosBias_, 0, gradSizeBytes));
		}
		AccumulateRelPosBiasGrad(attnGradWorkspace_, relPosIndex_, gradRelPosBias_, batchSize_, tokens_, numHeads_, relPosSize_, alphaWeights_);
	}
	PackHeadsToColumns(dQPacked_, dKPacked_, dVPacked_, dQ, dK, dV, batchSize_, tokens_, embedDim_, numHeads_);
	const long long dqdvdStrideA = static_cast<long long>(outNCHW_);
	const long long dqdvdStrideC = static_cast<long long>(embedDim_)*embedDim_;
	checkCLNN(CLNNGemmStridedBatchedEx(CLNN_OP_N, CLNN_OP_T, embedDim_, embedDim_, tokens_*batchSize_, &alphaWeights_, dQ, CUDA_R_16F, embedDim_, dqdvdStrideA, inData_, CUDA_R_16F, embedDim_, 0, betaWeights, gradQ_, CUDA_R_16F, embedDim_, dqdvdStrideC, 3, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, qWeights_, CUDA_R_16F, embedDim_, dQ, CUDA_R_16F, embedDim_, &zero_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, kWeights_, CUDA_R_16F, embedDim_, dK, CUDA_R_16F, embedDim_, &one_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, embedDim_, tokens_*batchSize_, embedDim_, &one_, vWeights_, CUDA_R_16F, embedDim_, dV, CUDA_R_16F, embedDim_, &one_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	return outGrad_;
}
void WmmaAttentionLayer::UpdateParameters(float lr){
	if(accumCount_ % gradAccumLength_ > 0) return;
	AdamWHalf(qWeights_, gradQ_, m_Q_, v_Q_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(kWeights_, gradK_, m_K_, v_K_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(vWeights_, gradV_, m_V_, v_V_, lr, t_, weightDecay_, embedDim_*embedDim_);
	AdamWHalf(oWeights_, gradOut_, m_O_, v_O_, lr, t_, weightDecay_, embedDim_*embedDim_);
	if(useRelPosBias_ && train_ && relPosBias_){
		const int biasElems = numHeads_ * relPosSize_;
		AdamWFloat(relPosBias_, gradRelPosBias_, m_relPosBias_, v_relPosBias_, lr, t_, weightDecay_, biasElems);
	}
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
	if(useRelPosBias_ && relPosBias_){
		const size_t biasSize = static_cast<size_t>(numHeads_) * relPosSize_ * sizeof(float);
		cudaMemcpy(buffer, relPosBias_, biasSize, cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<char*>(buffer), biasSize);
	}
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
	if(useRelPosBias_ && relPosBias_){
		const size_t biasSize = static_cast<size_t>(numHeads_) * relPosSize_ * sizeof(float);
		file.read(reinterpret_cast<char*>(buffer), biasSize);
		cudaMemcpy(relPosBias_, buffer, biasSize, cudaMemcpyHostToDevice);
	}
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
	if(useRelPosBias_ && relPosBias_){
		const size_t biasSize = static_cast<size_t>(numHeads_) * relPosSize_ * sizeof(float);
		cudaMemcpy(buffer, m_relPosBias_, biasSize, cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<char*>(buffer), biasSize);
		cudaMemcpy(buffer, v_relPosBias_, biasSize, cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<char*>(buffer), biasSize);
	}
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
	if(useRelPosBias_ && relPosBias_){
		const size_t biasSize = static_cast<size_t>(numHeads_) * relPosSize_ * sizeof(float);
		file.read(reinterpret_cast<char*>(buffer), biasSize);
		cudaMemcpy(m_relPosBias_, buffer, biasSize, cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), biasSize);
		cudaMemcpy(v_relPosBias_, buffer, biasSize, cudaMemcpyHostToDevice);
	}
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t WmmaAttentionLayer::GetParameterSize(){
	size_t size = 4*embedDim_*embedDim_*sizeof(__half);
	if(useRelPosBias_){ size += static_cast<size_t>(numHeads_) * relPosSize_ * sizeof(float); }
	return size;
}
size_t WmmaAttentionLayer::GetOptimizerStateSize(){
	if(!train_) return 0;
	size_t size = 8*embedDim_*embedDim_*sizeof(__half);
	if(useRelPosBias_){ size += 2*static_cast<size_t>(numHeads_) * relPosSize_ * sizeof(float); }
	return size + sizeof(int);
}
void WmmaAttentionLayer::SetTrain(bool enable){ train_ = enable; }

void WmmaAttentionLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!train_) return;
	const int matrixElems = embedDim_*embedDim_;
	halfTasks.push_back({qWeights_, gradQ_, m_Q_, v_Q_, matrixElems});
	halfTasks.push_back({kWeights_, gradK_, m_K_, v_K_, matrixElems});
	halfTasks.push_back({vWeights_, gradV_, m_V_, v_V_, matrixElems});
	halfTasks.push_back({oWeights_, gradOut_, m_O_, v_O_, matrixElems});
	if(useRelPosBias_ && relPosBias_){
		const int biasElems = numHeads_ * relPosSize_;
		floatTasks.push_back({relPosBias_, gradRelPosBias_, m_relPosBias_, v_relPosBias_, biasElems});
	}
}

void WmmaAttentionLayer::SetAttentionMask(const float* attentionMask, const int maskBatchSize, const int maskHeads){
	attentionMask_ = attentionMask;
	maskBatchSize_ = maskBatchSize;
	maskHeads_ = maskHeads;
}
void WmmaAttentionLayer::InitRelativePositionBias(int windowHeight, int windowWidth, const std::vector<int>& relPosIndex){
	if(windowHeight <= 0 || windowWidth <= 0){ throw std::invalid_argument("InitRelativePositionBias invalid window size"); }
	const size_t expectedSize = static_cast<size_t>(tokens_) * tokens_;
	if(relPosIndex.size() != expectedSize){ throw std::invalid_argument("InitRelativePositionBias relPosIndex size mismatch"); }
	relPosSize_ = (2*windowHeight - 1) * (2*windowWidth - 1);
	const size_t biasElems = static_cast<size_t>(numHeads_) * relPosSize_;
	if(relPosBias_ == nullptr){ CUDAMallocZero(&relPosBias_, biasElems * sizeof(float)); }
	if(relPosIndex_ == nullptr){ CUDAMallocZero(&relPosIndex_, expectedSize * sizeof(int)); }
	checkCUDA(cudaMemcpy(relPosIndex_, relPosIndex.data(), expectedSize * sizeof(int), cudaMemcpyHostToDevice));
	if(train_){
		if(gradRelPosBias_ == nullptr){ CUDAMallocZero(&gradRelPosBias_, biasElems * sizeof(float)); }
		if(m_relPosBias_ == nullptr){ CUDAMallocZero(&m_relPosBias_, biasElems * sizeof(float)); }
		if(v_relPosBias_ == nullptr){ CUDAMallocZero(&v_relPosBias_, biasElems * sizeof(float)); }
	}
	useRelPosBias_ = true;
}
