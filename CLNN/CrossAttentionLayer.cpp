#include "CrossAttentionLayer.h"
#include "HostCommon.h"
#include "CuCommon.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
CrossAttentionLayer::CrossAttentionLayer(const int batchSize, const int queryTokens, const int contextTokens, const int embedDim, const int numHeads, std::string layerName, const WeightInitMethod weightInitMethod, const int contextDim) :
	CrossAttentionLayer(batchSize, queryTokens, contextTokens, embedDim, numHeads, std::move(layerName), false, 0.0f, 1, weightInitMethod, contextDim){}
CrossAttentionLayer::CrossAttentionLayer(const int batchSize, const int queryTokens, const int contextTokens, const int embedDim, const int numHeads, std::string layerName, const bool train, const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod, const int contextDim) :
	batchSize_(batchSize), queryTokens_(queryTokens), contextTokens_(contextTokens), embedDim_(embedDim), contextDim_(contextDim > 0 ? contextDim : embedDim), numHeads_(numHeads), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	if(batchSize_ <= 0 || queryTokens_ <= 0 || contextTokens_ <= 0 || embedDim_ <= 0 || contextDim_ <= 0 || numHeads_ <= 0){
		throw std::invalid_argument("CrossAttentionLayer dimensions must be positive");
	}
	if(embedDim_ % numHeads_ != 0){ throw std::invalid_argument("CrossAttentionLayer embedDim must be divisible by numHeads"); }
	if(gradAccumLength_ <= 0){ throw std::invalid_argument("CrossAttentionLayer gradAccumLength must be positive"); }
	const size_t maxInt = static_cast<size_t>(std::numeric_limits<int>::max());
	queryElements_ = static_cast<size_t>(batchSize_)*queryTokens_*embedDim_;
	contextElements_ = static_cast<size_t>(batchSize_)*contextTokens_*embedDim_;
	contextInputElements_ = static_cast<size_t>(batchSize_)*contextTokens_*contextDim_;
	if(queryElements_ == 0 || contextElements_ == 0 || contextInputElements_ == 0 || queryElements_ > maxInt || contextElements_ > maxInt || contextInputElements_ > maxInt){
		throw std::overflow_error("CrossAttentionLayer tensor is too large");
	}
	const size_t queryColumns = static_cast<size_t>(batchSize_)*queryTokens_;
	const size_t contextColumns = static_cast<size_t>(batchSize_)*contextTokens_;
	if(queryColumns > maxInt || contextColumns > maxInt){ throw std::overflow_error("CrossAttentionLayer column count is too large"); }
	squareProjectionElements_ = static_cast<size_t>(embedDim_)*embedDim_;
	contextProjectionElements_ = static_cast<size_t>(embedDim_)*contextDim_;
	if(squareProjectionElements_ > maxInt || contextProjectionElements_ > maxInt){ throw std::overflow_error("CrossAttentionLayer projection is too large"); }
	const size_t attentionElements = static_cast<size_t>(batchSize_)*numHeads_*queryTokens_*contextTokens_;
	if(attentionElements > maxInt){ throw std::overflow_error("CrossAttentionLayer attention matrix is too large"); }
	layerName_ = std::move(layerName);
	train_ = train;
	headDim_ = embedDim_/numHeads_;
	outNCHW_ = queryElements_;
	weightCount_ = std::max(squareProjectionElements_, contextProjectionElements_);
	alphaWeights_ = 1.0f/(static_cast<float>(batchSize_)*queryTokens_*gradAccumLength_);
	CUDAMallocZero(&qWeights_, squareProjectionElements_*sizeof(__half));
	CUDAMallocZero(&kWeights_, contextProjectionElements_*sizeof(__half));
	CUDAMallocZero(&vWeights_, contextProjectionElements_*sizeof(__half));
	CUDAMallocZero(&oWeights_, squareProjectionElements_*sizeof(__half));
	WeightInit(qWeights_, static_cast<int>(squareProjectionElements_), embedDim_, embedDim_, weightInitMethod);
	WeightInit(kWeights_, static_cast<int>(contextProjectionElements_), contextDim_, embedDim_, weightInitMethod);
	WeightInit(vWeights_, static_cast<int>(contextProjectionElements_), contextDim_, embedDim_, weightInitMethod);
	WeightInit(oWeights_, static_cast<int>(squareProjectionElements_), embedDim_, embedDim_, weightInitMethod);
	CUDAMallocZero(&q_, queryElements_*sizeof(__half));
	CUDAMallocZero(&k_, contextElements_*sizeof(__half));
	CUDAMallocZero(&v_, contextElements_*sizeof(__half));
	CUDAMallocZero(&attnOut_, queryElements_*sizeof(__half));
	CUDAMallocZero(&qPacked_, queryElements_*sizeof(__half));
	CUDAMallocZero(&kPacked_, contextElements_*sizeof(__half));
	CUDAMallocZero(&vPacked_, contextElements_*sizeof(__half));
	CUDAMallocZero(&attnOutPacked_, queryElements_*sizeof(__half));
	CUDAMallocZero(&outData_, queryElements_*sizeof(__half));
	CUDAMallocZero(&attentionWeights_, attentionElements*sizeof(float));
	if(train_){ AllocateTrainingBuffers(); }
}
CrossAttentionLayer::~CrossAttentionLayer(){
	cudaFree(qWeights_);
	cudaFree(kWeights_);
	cudaFree(vWeights_);
	cudaFree(oWeights_);
	cudaFree(q_);
	cudaFree(k_);
	cudaFree(v_);
	cudaFree(attnOut_);
	cudaFree(qPacked_);
	cudaFree(kPacked_);
	cudaFree(vPacked_);
	cudaFree(attnOutPacked_);
	cudaFree(outData_);
	cudaFree(attentionWeights_);
	cudaFree(gradQWeights_);
	cudaFree(gradKWeights_);
	cudaFree(gradVWeights_);
	cudaFree(gradOWeights_);
	cudaFree(m_Q_);
	cudaFree(v_Q_);
	cudaFree(m_K_);
	cudaFree(v_K_);
	cudaFree(m_V_);
	cudaFree(v_V_);
	cudaFree(m_O_);
	cudaFree(v_O_);
	cudaFree(dQ_);
	cudaFree(dK_);
	cudaFree(dV_);
	cudaFree(dAttnOut_);
	cudaFree(dQPacked_);
	cudaFree(dKPacked_);
	cudaFree(dVPacked_);
	cudaFree(dAttnOutPacked_);
	cudaFree(outGrad_);
	cudaFree(contextGrad_);
	cudaFree(attentionGrad_);
}
void CrossAttentionLayer::AllocateTrainingBuffers(){
	if(trainingAllocated_) return;
	const size_t attentionElements = static_cast<size_t>(batchSize_)*numHeads_*queryTokens_*contextTokens_;
	CUDAMallocZero(&gradQWeights_, squareProjectionElements_*sizeof(__half));
	CUDAMallocZero(&gradKWeights_, contextProjectionElements_*sizeof(__half));
	CUDAMallocZero(&gradVWeights_, contextProjectionElements_*sizeof(__half));
	CUDAMallocZero(&gradOWeights_, squareProjectionElements_*sizeof(__half));
	CUDAMallocZero(&m_Q_, squareProjectionElements_*sizeof(__half));
	CUDAMallocZero(&v_Q_, squareProjectionElements_*sizeof(__half));
	CUDAMallocZero(&m_K_, contextProjectionElements_*sizeof(__half));
	CUDAMallocZero(&v_K_, contextProjectionElements_*sizeof(__half));
	CUDAMallocZero(&m_V_, contextProjectionElements_*sizeof(__half));
	CUDAMallocZero(&v_V_, contextProjectionElements_*sizeof(__half));
	CUDAMallocZero(&m_O_, squareProjectionElements_*sizeof(__half));
	CUDAMallocZero(&v_O_, squareProjectionElements_*sizeof(__half));
	CUDAMallocZero(&dQ_, queryElements_*sizeof(__half));
	CUDAMallocZero(&dK_, contextElements_*sizeof(__half));
	CUDAMallocZero(&dV_, contextElements_*sizeof(__half));
	CUDAMallocZero(&dAttnOut_, queryElements_*sizeof(__half));
	CUDAMallocZero(&dQPacked_, queryElements_*sizeof(__half));
	CUDAMallocZero(&dKPacked_, contextElements_*sizeof(__half));
	CUDAMallocZero(&dVPacked_, contextElements_*sizeof(__half));
	CUDAMallocZero(&dAttnOutPacked_, queryElements_*sizeof(__half));
	CUDAMallocZero(&outGrad_, queryElements_*sizeof(__half));
	CUDAMallocZero(&contextGrad_, contextInputElements_*sizeof(__half));
	CUDAMallocZero(&attentionGrad_, attentionElements*sizeof(float));
	trainingAllocated_ = true;
}
__half* CrossAttentionLayer::Forward(__half* data){
	const __half* context = contextData_;
	if(context == nullptr){
		if(contextTokens_ != queryTokens_ || contextDim_ != embedDim_){
			throw std::invalid_argument("CrossAttentionLayer::Forward needs SetContext or matching self-attention dimensions");
		}
		context = data;
	}
	return Forward(data, context);
}
__half* CrossAttentionLayer::Forward(__half* queryData, const __half* contextData){
	if(queryData == nullptr){ throw std::invalid_argument("CrossAttentionLayer::Forward received null query input"); }
	if(contextData == nullptr){ throw std::invalid_argument("CrossAttentionLayer::Forward received null context input"); }
	inData_ = queryData;
	contextInData_ = contextData;
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_*queryTokens_, embedDim_, &one_, qWeights_, CUDA_R_16F, embedDim_, queryData, CUDA_R_16F, embedDim_, &zero_, q_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_*contextTokens_, contextDim_, &one_, kWeights_, CUDA_R_16F, embedDim_, contextData, CUDA_R_16F, contextDim_, &zero_, k_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_*contextTokens_, contextDim_, &one_, vWeights_, CUDA_R_16F, embedDim_, contextData, CUDA_R_16F, contextDim_, &zero_, v_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	if(!PackColumnsToHeads(q_, qPacked_, batchSize_, queryTokens_, embedDim_, numHeads_)){
		throw std::runtime_error("CrossAttentionLayer query packing failed");
	}
	if(!PackColumnsToHeads(nullptr, k_, v_, nullptr, kPacked_, vPacked_, batchSize_, contextTokens_, embedDim_, numHeads_)){
		throw std::runtime_error("CrossAttentionLayer context packing failed");
	}
	if(!CrossAttentionForward(qPacked_, kPacked_, vPacked_, attnOutPacked_, attentionWeights_, attentionMask_, batchSize_, queryTokens_, contextTokens_, headDim_, numHeads_, maskBatchSize_, maskHeads_)){
		throw std::runtime_error("CrossAttentionLayer forward kernel failed");
	}
	if(!PackHeadsToColumns(attnOutPacked_, attnOut_, batchSize_, queryTokens_, embedDim_, numHeads_)){
		throw std::runtime_error("CrossAttentionLayer output unpacking failed");
	}
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_N, embedDim_, batchSize_*queryTokens_, embedDim_, &one_, oWeights_, CUDA_R_16F, embedDim_, attnOut_, CUDA_R_16F, embedDim_, &zero_, outData_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	return outData_;
}
__half* CrossAttentionLayer::Backward(__half* grad){
	if(grad == nullptr){ throw std::invalid_argument("CrossAttentionLayer::Backward received null gradient"); }
	if(!train_){ return grad; }
	if(!trainingAllocated_){ throw std::runtime_error("CrossAttentionLayer::Backward requires training buffers; construct with train=true"); }
	if(inData_ == nullptr || contextInData_ == nullptr){ throw std::runtime_error("CrossAttentionLayer::Backward requires a previous Forward call"); }
	const float* betaWeights = accumCount_++%gradAccumLength_ == 0 ? &zero_ : &one_;
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, embedDim_, batchSize_*queryTokens_, &alphaWeights_, grad, CUDA_R_16F, embedDim_, attnOut_, CUDA_R_16F, embedDim_, betaWeights, gradOWeights_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, embedDim_, batchSize_*queryTokens_, embedDim_, &one_, oWeights_, CUDA_R_16F, embedDim_, grad, CUDA_R_16F, embedDim_, &zero_, dAttnOut_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	if(!PackColumnsToHeads(dAttnOut_, dAttnOutPacked_, batchSize_, queryTokens_, embedDim_, numHeads_)){
		throw std::runtime_error("CrossAttentionLayer output-gradient packing failed");
	}
	if(!CrossAttentionBackward(qPacked_, kPacked_, vPacked_, dAttnOutPacked_, attentionWeights_, dQPacked_, dKPacked_, dVPacked_, attentionGrad_, batchSize_, queryTokens_, contextTokens_, headDim_, numHeads_)){
		throw std::runtime_error("CrossAttentionLayer backward kernel failed");
	}
	if(!PackHeadsToColumns(dQPacked_, dQ_, batchSize_, queryTokens_, embedDim_, numHeads_)){
		throw std::runtime_error("CrossAttentionLayer query-gradient unpacking failed");
	}
	if(!PackHeadsToColumns(nullptr, dKPacked_, dVPacked_, nullptr, dK_, dV_, batchSize_, contextTokens_, embedDim_, numHeads_)){
		throw std::runtime_error("CrossAttentionLayer context-gradient unpacking failed");
	}
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, embedDim_, batchSize_*queryTokens_, &alphaWeights_, dQ_, CUDA_R_16F, embedDim_, inData_, CUDA_R_16F, embedDim_, betaWeights, gradQWeights_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, contextDim_, batchSize_*contextTokens_, &alphaWeights_, dK_, CUDA_R_16F, embedDim_, contextInData_, CUDA_R_16F, contextDim_, betaWeights, gradKWeights_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_N, CLNN_OP_T, embedDim_, contextDim_, batchSize_*contextTokens_, &alphaWeights_, dV_, CUDA_R_16F, embedDim_, contextInData_, CUDA_R_16F, contextDim_, betaWeights, gradVWeights_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, embedDim_, batchSize_*queryTokens_, embedDim_, &one_, qWeights_, CUDA_R_16F, embedDim_, dQ_, CUDA_R_16F, embedDim_, &zero_, outGrad_, CUDA_R_16F, embedDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, contextDim_, batchSize_*contextTokens_, embedDim_, &one_, kWeights_, CUDA_R_16F, embedDim_, dK_, CUDA_R_16F, embedDim_, &zero_, contextGrad_, CUDA_R_16F, contextDim_, CUDA_R_32F));
	checkCLNN(CLNNGemmEx(CLNN_OP_T, CLNN_OP_N, contextDim_, batchSize_*contextTokens_, embedDim_, &one_, vWeights_, CUDA_R_16F, embedDim_, dV_, CUDA_R_16F, embedDim_, &one_, contextGrad_, CUDA_R_16F, contextDim_, CUDA_R_32F));
	return outGrad_;
}
void CrossAttentionLayer::UpdateParameters(const float learningRate){
	if(!train_ || !trainingAllocated_) return;
	if(accumCount_%gradAccumLength_ > 0) return;
	AdamWHalf(qWeights_, gradQWeights_, m_Q_, v_Q_, learningRate, t_, weightDecay_, static_cast<int>(squareProjectionElements_));
	AdamWHalf(kWeights_, gradKWeights_, m_K_, v_K_, learningRate, t_, weightDecay_, static_cast<int>(contextProjectionElements_));
	AdamWHalf(vWeights_, gradVWeights_, m_V_, v_V_, learningRate, t_, weightDecay_, static_cast<int>(contextProjectionElements_));
	AdamWHalf(oWeights_, gradOWeights_, m_O_, v_O_, learningRate, t_, weightDecay_, static_cast<int>(squareProjectionElements_));
	++t_;
}
void CrossAttentionLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	const size_t squareBytes = squareProjectionElements_*sizeof(__half);
	const size_t contextBytes = contextProjectionElements_*sizeof(__half);
	cudaMemcpy(buffer, qWeights_, squareBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), squareBytes);
	cudaMemcpy(buffer, kWeights_, contextBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), contextBytes);
	cudaMemcpy(buffer, vWeights_, contextBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), contextBytes);
	cudaMemcpy(buffer, oWeights_, squareBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), squareBytes);
}
void CrossAttentionLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	const size_t squareBytes = squareProjectionElements_*sizeof(__half);
	const size_t contextBytes = contextProjectionElements_*sizeof(__half);
	file.read(reinterpret_cast<char*>(buffer), squareBytes);
	cudaMemcpy(qWeights_, buffer, squareBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), contextBytes);
	cudaMemcpy(kWeights_, buffer, contextBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), contextBytes);
	cudaMemcpy(vWeights_, buffer, contextBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), squareBytes);
	cudaMemcpy(oWeights_, buffer, squareBytes, cudaMemcpyHostToDevice);
}
void CrossAttentionLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!trainingAllocated_) return;
	const size_t squareBytes = squareProjectionElements_*sizeof(__half);
	const size_t contextBytes = contextProjectionElements_*sizeof(__half);
	cudaMemcpy(buffer, m_Q_, squareBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), squareBytes);
	cudaMemcpy(buffer, v_Q_, squareBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), squareBytes);
	cudaMemcpy(buffer, m_K_, contextBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), contextBytes);
	cudaMemcpy(buffer, v_K_, contextBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), contextBytes);
	cudaMemcpy(buffer, m_V_, contextBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), contextBytes);
	cudaMemcpy(buffer, v_V_, contextBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), contextBytes);
	cudaMemcpy(buffer, m_O_, squareBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), squareBytes);
	cudaMemcpy(buffer, v_O_, squareBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), squareBytes);
	file.write(reinterpret_cast<const char*>(&t_), sizeof(int));
}
void CrossAttentionLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!trainingAllocated_){ return; }
	const size_t squareBytes = squareProjectionElements_*sizeof(__half);
	const size_t contextBytes = contextProjectionElements_*sizeof(__half);
	file.read(reinterpret_cast<char*>(buffer), squareBytes);
	cudaMemcpy(m_Q_, buffer, squareBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), squareBytes);
	cudaMemcpy(v_Q_, buffer, squareBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), contextBytes);
	cudaMemcpy(m_K_, buffer, contextBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), contextBytes);
	cudaMemcpy(v_K_, buffer, contextBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), contextBytes);
	cudaMemcpy(m_V_, buffer, contextBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), contextBytes);
	cudaMemcpy(v_V_, buffer, contextBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), squareBytes);
	cudaMemcpy(m_O_, buffer, squareBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), squareBytes);
	cudaMemcpy(v_O_, buffer, squareBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(&t_), sizeof(int));
}
size_t CrossAttentionLayer::GetParameterSize(){
	return weightCount_*sizeof(__half);
}
size_t CrossAttentionLayer::GetOptimizerStateSize(){
	return trainingAllocated_ ? std::max(weightCount_*sizeof(__half), static_cast<size_t>(sizeof(int))) : 0;
}
void CrossAttentionLayer::SetTrain(const bool enable){
	if(enable && !trainingAllocated_){ throw std::runtime_error("CrossAttentionLayer cannot enable training when constructed for inference"); }
	train_ = enable;
}
void CrossAttentionLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	if(!train_ || !trainingAllocated_) return;
	halfTasks.push_back({qWeights_, gradQWeights_, m_Q_, v_Q_, static_cast<int>(squareProjectionElements_), weightDecay_});
	halfTasks.push_back({kWeights_, gradKWeights_, m_K_, v_K_, static_cast<int>(contextProjectionElements_), weightDecay_});
	halfTasks.push_back({vWeights_, gradVWeights_, m_V_, v_V_, static_cast<int>(contextProjectionElements_), weightDecay_});
	halfTasks.push_back({oWeights_, gradOWeights_, m_O_, v_O_, static_cast<int>(squareProjectionElements_), weightDecay_});
}
void CrossAttentionLayer::SetContext(const __half* contextData){
	contextData_ = contextData;
}
void CrossAttentionLayer::ClearContext(){
	contextData_ = nullptr;
}
void CrossAttentionLayer::SetAttentionMask(const float* attentionMask, const int maskBatchSize, const int maskHeads){
	if(attentionMask != nullptr && (maskBatchSize <= 0 || maskHeads <= 0)){
		throw std::invalid_argument("CrossAttentionLayer attention mask dimensions must be positive");
	}
	attentionMask_ = attentionMask;
	maskBatchSize_ = maskBatchSize;
	maskHeads_ = maskHeads;
}
float* CrossAttentionLayer::GetAttentionWeights() const{
	return attentionWeights_;
}
__half* CrossAttentionLayer::GetContextGrad() const{
	return contextGrad_;
}
