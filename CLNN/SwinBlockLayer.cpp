#include "SwinBlockLayer.h"
#include "CuCommon.cuh"
#include "Dropout.h"
#include "DropPath.h"
#include "FCLayer.h"
#include "GELULayer.h"
#include "LayerNorm.h"
#include "WmmaAttentionLayer.h"
#include <algorithm>
#include <vector>
SwinBlockLayer::SwinBlockLayer(const cudnnHandle_t cudnnHandle, const int batchSize, const int nTokens, const int embedDim, const int ffDim, const int numHeads, const int patchRows, const int patchCols, const int windowHeight, const int windowWidth, const int shiftHeight, const int shiftWidth, const float dropPathRate, std::string layerName, const bool train, const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod, __half* windowedInput, __half* windowedGrad, __half* tokens, float* sharedAttentionMask, const bool ownsAttentionMask, __half* attentionWorkspace, __half* qPacked, __half* kPacked, __half* vPacked, __half* attnOutPacked, __half* dQPacked, __half* dKPacked, __half* dVPacked, float* attnGradWorkspace) : cudnnHandle_(cudnnHandle), batchSize_(batchSize), nTokens_(nTokens), embedDim_(embedDim), ffDim_(ffDim), numHeads_(numHeads), patchRows_(patchRows), patchCols_(patchCols), windowHeight_(windowHeight), windowWidth_(windowWidth), shiftHeight_(shiftHeight), shiftWidth_(shiftWidth){
	layerName_ = layerName;
	train_ = train;
	if(nTokens_ != patchRows_ * patchCols_){ throw std::invalid_argument("SwinBlockLayer tokens must match patch grid"); }
	if(patchRows_ % windowHeight_ != 0 || patchCols_ % windowWidth_ != 0){ throw std::invalid_argument("SwinBlockLayer window size must evenly divide patch rows/cols"); }
	windowTokens_ = windowHeight_ * windowWidth_;
	windowCount_ = (patchRows_ / windowHeight_) * (patchCols_ / windowWidth_);
	windowBatch_ = batchSize_ * windowCount_;
	outNCHW_ = batchSize_ * nTokens_ * embedDim_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_*nTokens_, embedDim_, 1, 1));
	norm1_ = new LayerNorm(batchSize_ * nTokens_, embedDim_, 1, 1, "SwinNorm1", train);
	layers_.push_back(norm1_);
	attention_ = new WmmaAttentionLayer(cudnnHandle_, windowBatch_, windowTokens_, embedDim_, numHeads_, "SwinAttention", train, weightDecay, gradAccumLength, weightInitMethod, attentionWorkspace, qPacked, kPacked, vPacked, attnOutPacked, dQPacked, dKPacked, dVPacked, attnGradWorkspace);
	layers_.push_back(attention_);
	attnDrop_ = new Dropout(cudnnHandle_, 0.1f, batchSize_ * nTokens_, embedDim_, 1, 1, "SwinAttnDropout", train);
	layers_.push_back(attnDrop_);
	attnDropPath_ = new DropPath(dropPathRate, batchSize_, nTokens_ * embedDim_, "SwinAttnDropPath", train);
	layers_.push_back(attnDropPath_);
	norm2_ = new LayerNorm(batchSize_ * nTokens_, embedDim_, 1, 1, "SwinNorm2", train);
	layers_.push_back(norm2_);
	fc1_ = new FCLayer(batchSize_ * nTokens_, embedDim_, ffDim_, "SwinFC1", train, weightDecay, gradAccumLength, weightInitMethod);
	layers_.push_back(fc1_);
	gelu_ = new GELULayer(batchSize_ * nTokens_, ffDim_, 1, 1, "SwinGELU");
	layers_.push_back(gelu_);
	fc2_ = new FCLayer(batchSize_ * nTokens_, ffDim_, embedDim_, "SwinFC2", train, weightDecay, gradAccumLength, weightInitMethod);
	layers_.push_back(fc2_);
	ffDrop_ = new Dropout(cudnnHandle_, 0.1f, batchSize_ * nTokens_, embedDim_, 1, 1, "SwinFfDropout", train);
	layers_.push_back(ffDrop_);
	ffDropPath_ = new DropPath(dropPathRate, batchSize_, nTokens_ * embedDim_, "SwinFfDropPath", train);
	layers_.push_back(ffDropPath_);
	{
		std::vector<int> relPosIndex(windowTokens_ * windowTokens_, 0);
		const int relWidth = 2*windowWidth_ - 1;
		for(int i = 0; i < windowTokens_; ++i){
			const int rowI = i / windowWidth_;
			const int colI = i % windowWidth_;
			for(int j = 0; j < windowTokens_; ++j){
				const int rowJ = j / windowWidth_;
				const int colJ = j % windowWidth_;
				const int dy = rowI - rowJ + windowHeight_ - 1;
				const int dx = colI - colJ + windowWidth_ - 1;
				relPosIndex[i * windowTokens_ + j] = dy * relWidth + dx;
			}
		}
		attention_->InitRelativePositionBias(windowHeight_, windowWidth_, relPosIndex);
	}
	attentionMask_ = sharedAttentionMask;
	ownsAttentionMask_ = ownsAttentionMask;
	if(attentionMask_ == nullptr && (shiftHeight_ > 0 || shiftWidth_ > 0)){
		const int windowsCols = patchCols_ / windowWidth_;
		constexpr float maskValue = -1e4f;
		const size_t windowMaskSize = static_cast<size_t>(windowCount_) * windowTokens_ * windowTokens_;
		std::vector<float> hostMask(windowMaskSize, 0.0f);
		std::vector<int> tokenWindowIds(windowTokens_, 0);
		for(int windowIndex = 0; windowIndex < windowCount_; ++windowIndex){
			const int windowRow = windowIndex / windowsCols;
			const int windowCol = windowIndex % windowsCols;
			for(int token = 0; token < windowTokens_; ++token){
				const int localRow = token / windowWidth_;
				const int localCol = token % windowWidth_;
				const int shiftedRow = windowRow * windowHeight_ + localRow;
				const int shiftedCol = windowCol * windowWidth_ + localCol;
				const int origRow = (shiftedRow - shiftHeight_ + patchRows_) % patchRows_;
				const int origCol = (shiftedCol - shiftWidth_ + patchCols_) % patchCols_;
				const int origWindowRow = origRow / windowHeight_;
				const int origWindowCol = origCol / windowWidth_;
				tokenWindowIds[token] = origWindowRow * windowsCols + origWindowCol;
			}
			for(int i = 0; i < windowTokens_; ++i){
				const size_t base = (static_cast<size_t>(windowIndex) * windowTokens_ + i) * windowTokens_;
				for(int j = 0; j < windowTokens_; ++j){
					if(tokenWindowIds[i] != tokenWindowIds[j]){ hostMask[base + j] = maskValue; }
				}
			}
		}
		CUDAMallocZero(&attentionMask_, windowMaskSize * sizeof(float));
		checkCUDA(cudaMemcpy(attentionMask_, hostMask.data(), windowMaskSize*sizeof(float), cudaMemcpyHostToDevice));
		ownsAttentionMask_ = true;
	}
	attention_->SetAttentionMask(attentionMask_, windowCount_, 1);
	windowedInput_ = windowedInput;
	windowedGrad_ = windowedGrad;
	tokens_ = tokens;
	if(windowedInput_ == nullptr || windowedGrad_ == nullptr || tokens_ == nullptr){
		ownsWorkspace_ = true;
		const size_t windowElems = static_cast<size_t>(windowBatch_) * windowTokens_ * embedDim_;
		CUDAMallocZero(&windowedInput_, windowElems * sizeof(__half));
		CUDAMallocZero(&windowedGrad_, windowElems * sizeof(__half));
		CUDAMallocZero(&tokens_, static_cast<size_t>(batchSize_) * nTokens_ * embedDim_ * sizeof(__half));
	} else{
		ownsWorkspace_ = false;
	}
}
SwinBlockLayer::~SwinBlockLayer(){
	for(const auto layer : layers_) delete layer;
	layers_.clear();
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	if(ownsWorkspace_){
		cudaFree(windowedInput_);
		cudaFree(windowedGrad_);
		cudaFree(tokens_);
	}
	if(ownsAttentionMask_){ cudaFree(attentionMask_); }
}
__half* SwinBlockLayer::Forward(__half* data){
	const auto* residual1 = data;
	data = norm1_->Forward(data);
	TokensToWindows(data, windowedInput_, batchSize_, nTokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	data = attention_->Forward(windowedInput_);
	WindowsToTokens(data, tokens_, batchSize_, nTokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	data = attnDrop_->Forward(tokens_);
	data = attnDropPath_->Forward(data);
	AddTensor(mixFwd_, data, mixFwd_, residual1, static_cast<int>(outNCHW_));
	const auto* residual2 = data;
	data = norm2_->Forward(data);
	data = fc1_->Forward(data);
	data = gelu_->Forward(data);
	data = fc2_->Forward(data);
	data = ffDrop_->Forward(data);
	data = ffDropPath_->Forward(data);
	AddTensor(mixFwd_, data, mixFwd_, residual2, static_cast<int>(outNCHW_));
	return data;
}
__half* SwinBlockLayer::Backward(__half* grad){
	const auto* residual2 = grad;
	grad = ffDropPath_->Backward(grad);
	grad = ffDrop_->Backward(grad);
	grad = fc2_->Backward(grad);
	grad = gelu_->Backward(grad);
	grad = fc1_->Backward(grad);
	grad = norm2_->Backward(grad);
	AddTensor(mixBwd_, grad, mixBwd_, residual2, static_cast<int>(outNCHW_));
	const auto* residual1 = grad;
	grad = attnDropPath_->Backward(grad);
	grad = attnDrop_->Backward(grad);
	TokensToWindows(grad, windowedGrad_, batchSize_, nTokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	grad = attention_->Backward(windowedGrad_);
	WindowsToTokens(grad, tokens_, batchSize_, nTokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	grad = norm1_->Backward(tokens_);
	AddTensor(mixBwd_, grad, mixBwd_, residual1, static_cast<int>(outNCHW_));
	return grad;
}
void SwinBlockLayer::UpdateParameters(const float lr){
	for(const auto layer : layers_){ layer->UpdateParameters(lr); }
}
void SwinBlockLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(const auto layer : layers_){ layer->SaveParameters(file, buffer); }
}
void SwinBlockLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(const auto layer : layers_){ layer->LoadParameters(file, buffer); }
}
void SwinBlockLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(const auto layer : layers_){ layer->SaveOptimizerState(file, buffer); }
}
void SwinBlockLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(const auto layer : layers_){ layer->LoadOptimizerState(file, buffer); }
}
size_t SwinBlockLayer::GetParameterSize(){
	size_t maxSize = 0;
	for(const auto layer : layers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	return maxSize;
}
size_t SwinBlockLayer::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(const auto layer : layers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	return maxSize;
}
void SwinBlockLayer::SetTrain(const bool enable){
	train_ = enable;
	for(const auto layer : layers_){ layer->SetTrain(enable); }
}

void SwinBlockLayer::CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks){
	for(const auto layer : layers_){ layer->CollectAdamWTasks(halfTasks, floatTasks); }
}

