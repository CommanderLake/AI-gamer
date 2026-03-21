#include "TemporalMemoryLayer.h"
#include "NNCommon.h"
#include <algorithm>
#include <stdexcept>
TemporalMemoryLayer::TemporalMemoryLayer(const int batchSize, const int embedDim, const int numHeads, const int maxContext, std::string layerName, const bool train, const float weightDecay, const int gradAccumLength) : batchSize_(batchSize), embedDim_(embedDim), numHeads_(numHeads), headDim_(embedDim / numHeads), maxContext_(maxContext), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	if(batchSize_ <= 0 || embedDim_ <= 0 || numHeads_ <= 0 || maxContext_ <= 0){ throw std::invalid_argument("TemporalMemoryLayer invalid dimensions"); }
	if(embedDim_ % numHeads_ != 0){ throw std::invalid_argument("TemporalMemoryLayer embedDim must be divisible by numHeads"); }
	layerName_ = std::move(layerName);
	train_ = train;
	outNCHW_ = static_cast<size_t>(batchSize_) * embedDim_;
	alphaWeights_ = 1.0f / static_cast<float>(batchSize_ * std::max(1, gradAccumLength_));
	const size_t projElems = static_cast<size_t>(embedDim_) * embedDim_;
	const size_t memoryElems = static_cast<size_t>(batchSize_) * maxContext_ * embedDim_;
	const size_t packedMemoryElems = memoryElems;
	const size_t qElems = static_cast<size_t>(batchSize_) * embedDim_;
	const size_t packedQElems = qElems;
	const size_t attnElems = static_cast<size_t>(batchSize_) * numHeads_ * maxContext_;
	CUDAMallocZero(&qWeights_, projElems * sizeof(__half));
	CUDAMallocZero(&kWeights_, projElems * sizeof(__half));
	CUDAMallocZero(&vWeights_, projElems * sizeof(__half));
	CUDAMallocZero(&oWeights_, projElems * sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_ * sizeof(__half));
	CUDAMallocZero(&qProj_, qElems * sizeof(__half));
	CUDAMallocZero(&kProj_, memoryElems * sizeof(__half));
	CUDAMallocZero(&vProj_, memoryElems * sizeof(__half));
	CUDAMallocZero(&context_, qElems * sizeof(__half));
	CUDAMallocZero(&qPacked_, packedQElems * sizeof(__half));
	CUDAMallocZero(&kPacked_, packedMemoryElems * sizeof(__half));
	CUDAMallocZero(&vPacked_, packedMemoryElems * sizeof(__half));
	CUDAMallocZero(&contextPacked_, packedQElems * sizeof(__half));
	CUDAMallocZero(&attnWeights_, attnElems * sizeof(float));
	if(train_){
		WeightInit(qWeights_, static_cast<int>(projElems), embedDim_, embedDim_, Xavier);
		WeightInit(kWeights_, static_cast<int>(projElems), embedDim_, embedDim_, Xavier);
		WeightInit(vWeights_, static_cast<int>(projElems), embedDim_, embedDim_, Xavier);
		WeightInit(oWeights_, static_cast<int>(projElems), embedDim_, embedDim_, Xavier);
		CUDAMallocZero(&gradQ_, projElems * sizeof(__half));
		CUDAMallocZero(&gradK_, projElems * sizeof(__half));
		CUDAMallocZero(&gradV_, projElems * sizeof(__half));
		CUDAMallocZero(&gradO_, projElems * sizeof(__half));
		CUDAMallocZero(&m_Q_, projElems * sizeof(__half));
		CUDAMallocZero(&v_Q_, projElems * sizeof(__half));
		CUDAMallocZero(&m_K_, projElems * sizeof(__half));
		CUDAMallocZero(&v_K_, projElems * sizeof(__half));
		CUDAMallocZero(&m_V_, projElems * sizeof(__half));
		CUDAMallocZero(&v_V_, projElems * sizeof(__half));
		CUDAMallocZero(&m_O_, projElems * sizeof(__half));
		CUDAMallocZero(&v_O_, projElems * sizeof(__half));
		CUDAMallocZero(&outGrad_, outNCHW_ * sizeof(__half));
		CUDAMallocZero(&gradQPacked_, packedQElems * sizeof(__half));
		CUDAMallocZero(&gradKPacked_, packedMemoryElems * sizeof(__half));
		CUDAMallocZero(&gradVPacked_, packedMemoryElems * sizeof(__half));
		CUDAMallocZero(&gradScores_, attnElems * sizeof(float));
	}
}
TemporalMemoryLayer::~TemporalMemoryLayer(){
	cudaFree(qWeights_);
	cudaFree(kWeights_);
	cudaFree(vWeights_);
	cudaFree(oWeights_);
	cudaFree(outData_);
	cudaFree(qProj_);
	cudaFree(kProj_);
	cudaFree(vProj_);
	cudaFree(context_);
	cudaFree(qPacked_);
	cudaFree(kPacked_);
	cudaFree(vPacked_);
	cudaFree(contextPacked_);
	cudaFree(attnWeights_);
	cudaFree(gradQ_);
	cudaFree(gradK_);
	cudaFree(gradV_);
	cudaFree(gradO_);
	cudaFree(m_Q_);
	cudaFree(v_Q_);
	cudaFree(m_K_);
	cudaFree(v_K_);
	cudaFree(m_V_);
	cudaFree(v_V_);
	cudaFree(m_O_);
	cudaFree(v_O_);
	cudaFree(outGrad_);
	cudaFree(gradQPacked_);
	cudaFree(gradKPacked_);
	cudaFree(gradVPacked_);
	cudaFree(gradScores_);
}
__half* TemporalMemoryLayer::Forward(__half* data){
	inData_ = data;
	if(memory_ == nullptr || validCounts_ == nullptr){ checkCUDA(cudaMemcpy(outData_, data, outNCHW_ * sizeof(__half), cudaMemcpyDeviceToDevice)); return outData_; }
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_, embedDim_, &one_, qWeights_, CUDA_R_16F, embedDim_, data, CUDA_R_16F, embedDim_, &zero_, qProj_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_ * maxContext_, embedDim_, &one_, kWeights_, CUDA_R_16F, embedDim_, memory_, CUDA_R_16F, embedDim_, &zero_, kProj_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_ * maxContext_, embedDim_, &one_, vWeights_, CUDA_R_16F, embedDim_, memory_, CUDA_R_16F, embedDim_, &zero_, vProj_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	PackColumnsToHeads(qProj_, qPacked_, batchSize_, 1, embedDim_, numHeads_);
	PackColumnsToHeads(kProj_, kPacked_, batchSize_, maxContext_, embedDim_, numHeads_);
	PackColumnsToHeads(vProj_, vPacked_, batchSize_, maxContext_, embedDim_, numHeads_);
	TemporalMemoryForward(qPacked_, kPacked_, vPacked_, validCounts_, attnWeights_, contextPacked_, batchSize_, maxContext_, numHeads_, headDim_);
	PackHeadsToColumns(contextPacked_, context_, batchSize_, 1, embedDim_, numHeads_);
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_, embedDim_, &one_, oWeights_, CUDA_R_16F, embedDim_, context_, CUDA_R_16F, embedDim_, &zero_, outData_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	return outData_;
}
__half* TemporalMemoryLayer::Backward(__half* grad){
	if(memory_ == nullptr || validCounts_ == nullptr){ checkCUDA(cudaMemcpy(outGrad_, grad, outNCHW_ * sizeof(__half), cudaMemcpyDeviceToDevice)); return outGrad_; }
	const float* betaWeights = accumCount_++ % gradAccumLength_ == 0 ? &zero_ : &one_;
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, embedDim_, batchSize_, &alphaWeights_, grad, CUDA_R_16F, embedDim_, context_, CUDA_R_16F, embedDim_, betaWeights, gradO_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, embedDim_, batchSize_, embedDim_, &one_, oWeights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &zero_, context_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	PackColumnsToHeads(context_, contextPacked_, batchSize_, 1, embedDim_, numHeads_);
	TemporalMemoryBackward(qPacked_, kPacked_, vPacked_, contextPacked_, attnWeights_, validCounts_, gradQPacked_, gradKPacked_, gradVPacked_, gradScores_, batchSize_, maxContext_, numHeads_, headDim_);
	PackHeadsToColumns(gradQPacked_, qProj_, batchSize_, 1, embedDim_, numHeads_);
	PackHeadsToColumns(gradKPacked_, kProj_, batchSize_, maxContext_, embedDim_, numHeads_);
	PackHeadsToColumns(gradVPacked_, vProj_, batchSize_, maxContext_, embedDim_, numHeads_);
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, embedDim_, batchSize_, &alphaWeights_, qProj_, CUDA_R_16F, embedDim_, inData_, CUDA_R_16F, embedDim_, betaWeights, gradQ_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, embedDim_, batchSize_ * maxContext_, &alphaWeights_, kProj_, CUDA_R_16F, embedDim_, memory_, CUDA_R_16F, embedDim_, betaWeights, gradK_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, embedDim_, batchSize_ * maxContext_, &alphaWeights_, vProj_, CUDA_R_16F, embedDim_, memory_, CUDA_R_16F, embedDim_, betaWeights, gradV_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, embedDim_, batchSize_, embedDim_, &one_, qWeights_, CUDA_R_16F, embedDim_, qProj_, CUDA_R_16F, embedDim_, &zero_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	return outGrad_;
}
void TemporalMemoryLayer::UpdateParameters(const float lr){
	if(accumCount_ % gradAccumLength_ > 0) return;
	const int matrixElems = embedDim_ * embedDim_;
	AdamWHalf(qWeights_, gradQ_, m_Q_, v_Q_, lr, t_, weightDecay_, matrixElems);
	AdamWHalf(kWeights_, gradK_, m_K_, v_K_, lr, t_, weightDecay_, matrixElems);
	AdamWHalf(vWeights_, gradV_, m_V_, v_V_, lr, t_, weightDecay_, matrixElems);
	AdamWHalf(oWeights_, gradO_, m_O_, v_O_, lr, t_, weightDecay_, matrixElems);
	++t_;
}
void TemporalMemoryLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	const size_t paramSize = static_cast<size_t>(embedDim_) * embedDim_ * sizeof(__half);
	cudaMemcpy(buffer, qWeights_, paramSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(buffer, kWeights_, paramSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(buffer, vWeights_, paramSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), paramSize);
	cudaMemcpy(buffer, oWeights_, paramSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), paramSize);
}
void TemporalMemoryLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	const size_t paramSize = static_cast<size_t>(embedDim_) * embedDim_ * sizeof(__half);
	if(!file.read(reinterpret_cast<char*>(buffer), paramSize)){ file.clear(); return; } cudaMemcpy(qWeights_, buffer, paramSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), paramSize)){ file.clear(); return; } cudaMemcpy(kWeights_, buffer, paramSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), paramSize)){ file.clear(); return; } cudaMemcpy(vWeights_, buffer, paramSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), paramSize)){ file.clear(); return; } cudaMemcpy(oWeights_, buffer, paramSize, cudaMemcpyHostToDevice);
}
void TemporalMemoryLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!train_) return;
	const size_t stateSize = static_cast<size_t>(embedDim_) * embedDim_ * sizeof(__half);
	cudaMemcpy(buffer, m_Q_, stateSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, m_K_, stateSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, m_V_, stateSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, m_O_, stateSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, v_Q_, stateSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, v_K_, stateSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, v_V_, stateSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), stateSize);
	cudaMemcpy(buffer, v_O_, stateSize, cudaMemcpyDeviceToHost); file.write(reinterpret_cast<char*>(buffer), stateSize);
	file.write(reinterpret_cast<char*>(&t_), sizeof(t_));
}
void TemporalMemoryLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!train_) return;
	const size_t stateSize = static_cast<size_t>(embedDim_) * embedDim_ * sizeof(__half);
	if(!file.read(reinterpret_cast<char*>(buffer), stateSize)){ file.clear(); return; } cudaMemcpy(m_Q_, buffer, stateSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), stateSize)){ file.clear(); return; } cudaMemcpy(m_K_, buffer, stateSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), stateSize)){ file.clear(); return; } cudaMemcpy(m_V_, buffer, stateSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), stateSize)){ file.clear(); return; } cudaMemcpy(m_O_, buffer, stateSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), stateSize)){ file.clear(); return; } cudaMemcpy(v_Q_, buffer, stateSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), stateSize)){ file.clear(); return; } cudaMemcpy(v_K_, buffer, stateSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), stateSize)){ file.clear(); return; } cudaMemcpy(v_V_, buffer, stateSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(buffer), stateSize)){ file.clear(); return; } cudaMemcpy(v_O_, buffer, stateSize, cudaMemcpyHostToDevice);
	if(!file.read(reinterpret_cast<char*>(&t_), sizeof(t_))){ file.clear(); return; }
}
size_t TemporalMemoryLayer::GetParameterSize(){ return 4ULL * embedDim_ * embedDim_ * sizeof(__half); }
size_t TemporalMemoryLayer::GetOptimizerStateSize(){ return train_ ? 8ULL * embedDim_ * embedDim_ * sizeof(__half) + sizeof(t_) : 0; }
void TemporalMemoryLayer::SetTrain(const bool enable){ train_ = enable; }
void TemporalMemoryLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!train_) return;
	const int matrixElems = embedDim_ * embedDim_;
	halfTasks.push_back({qWeights_, gradQ_, m_Q_, v_Q_, matrixElems, weightDecay_});
	halfTasks.push_back({kWeights_, gradK_, m_K_, v_K_, matrixElems, weightDecay_});
	halfTasks.push_back({vWeights_, gradV_, m_V_, v_V_, matrixElems, weightDecay_});
	halfTasks.push_back({oWeights_, gradO_, m_O_, v_O_, matrixElems, weightDecay_});
}
void TemporalMemoryLayer::SetMemory(const __half* memory, const int* validCounts){ memory_ = memory; validCounts_ = validCounts; }
