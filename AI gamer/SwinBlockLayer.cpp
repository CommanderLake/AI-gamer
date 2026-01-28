#include "SwinBlockLayer.h"
#include "CuCommon.cuh"
#include "Dropout.h"
#include "FCLayer.h"
#include "GELULayer.h"
#include "LayerNorm.h"
#include "WmmaAttentionLayer.h"
#include <algorithm>
#include <cmath>
#include <vector>
SwinBlockLayer::SwinBlockLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int tokens, const int embedDim, const int ffDim, const int numHeads, const int patchRows, const int patchCols, const int windowHeight, const int windowWidth,
								const int shiftHeight, const int shiftWidth, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), batchSize_(batchSize),
	tokens_(tokens), embedDim_(embedDim), ffDim_(ffDim), numHeads_(numHeads), patchRows_(patchRows), patchCols_(patchCols), windowHeight_(windowHeight), windowWidth_(windowWidth), shiftHeight_(shiftHeight), shiftWidth_(shiftWidth){
	layerName_ = layerName;
	train_ = train;
	if(tokens_ != patchRows_ * patchCols_){ throw std::invalid_argument("SwinBlockLayer tokens must match patch grid"); }
	if(patchRows_ % windowHeight_ != 0 || patchCols_ % windowWidth_ != 0){ throw std::invalid_argument("SwinBlockLayer window size must evenly divide patch rows/cols"); }
	windowTokens_ = windowHeight_ * windowWidth_;
	windowCount_ = (patchRows_ / windowHeight_) * (patchCols_ / windowWidth_);
	windowBatch_ = batchSize_ * windowCount_;
	outNCHW_ = batchSize_ * tokens_ * embedDim_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_*tokens_, embedDim_, 1, 1));
	norm1_ = new LayerNorm(batchSize_ * tokens_, embedDim_, 1, 1, "SwinNorm1", train);
	layers_.push_back(norm1_);
	attention_ = new WmmaAttentionLayer(cudnnHandle_, cublasHandle_, windowBatch_, windowTokens_, embedDim_, numHeads_, "SwinAttention", train, weightDecay, gradAccumLength, weightInitMethod);
	layers_.push_back(attention_);
	attnDrop_ = new Dropout(cudnnHandle_, 0.1f, batchSize_ * tokens_, embedDim_, 1, 1, "SwinAttnDropout", train);
	layers_.push_back(attnDrop_);
	norm2_ = new LayerNorm(batchSize_ * tokens_, embedDim_, 1, 1, "SwinNorm2", train);
	layers_.push_back(norm2_);
	fc1_ = new FCLayer(cublasHandle_, batchSize_ * tokens_, embedDim_, ffDim_, "SwinFC1", train, weightDecay, gradAccumLength, weightInitMethod);
	layers_.push_back(fc1_);
	gelu_ = new GELULayer(batchSize_ * tokens_, ffDim_, 1, 1, "SwinGELU");
	layers_.push_back(gelu_);
	fc2_ = new FCLayer(cublasHandle_, batchSize_ * tokens_, ffDim_, embedDim_, "SwinFC2", train, weightDecay, gradAccumLength, weightInitMethod);
	layers_.push_back(fc2_);
	ffDrop_ = new Dropout(cudnnHandle_, 0.1f, batchSize_ * tokens_, embedDim_, 1, 1, "SwinFfDropout", train);
	layers_.push_back(ffDrop_);
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
	{
		const int windowsCols = patchCols_ / windowWidth_;
		constexpr float maskValue = -1e4f;
		if(shiftHeight_ > 0 || shiftWidth_ > 0){
			const size_t windowMaskSize = static_cast<size_t>(windowBatch_) * numHeads_ * windowTokens_ * windowTokens_;
			std::vector<float> hostMask(windowMaskSize, 0.0f);
			std::vector<int> tokenWindowIds(windowTokens_, 0);
			for(int batch = 0; batch < batchSize_; ++batch){
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
					for(int head = 0; head < numHeads_; ++head){
						for(int i = 0; i < windowTokens_; ++i){
							const size_t base = ((static_cast<size_t>(batch) * windowCount_ + windowIndex) * numHeads_ + head) * windowTokens_ * windowTokens_ + static_cast<size_t>(i) * windowTokens_;
							for(int j = 0; j < windowTokens_; ++j){
								if(tokenWindowIds[i] != tokenWindowIds[j]){ hostMask[base + j] = maskValue; }
							}
						}
					}
				}
			}
			CUDAMallocZero(&attentionMask_, windowMaskSize * sizeof(float));
			checkCUDA(cudaMemcpy(attentionMask_, hostMask.data(), windowMaskSize*sizeof(float), cudaMemcpyHostToDevice));
		}
	}
	attention_->SetAttentionMask(attentionMask_);
	const size_t windowElems = static_cast<size_t>(windowBatch_) * windowTokens_ * embedDim_;
	CUDAMallocZero(&windowedInput_, windowElems * sizeof(__half));
	CUDAMallocZero(&windowedGrad_, windowElems * sizeof(__half));
	CUDAMallocZero(&tokenBuffer_, static_cast<size_t>(batchSize_) * tokens_ * embedDim_ * sizeof(__half));
}
SwinBlockLayer::~SwinBlockLayer(){
	for(const auto layer : layers_) delete layer;
	layers_.clear();
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	cudaFree(windowedInput_);
	cudaFree(windowedGrad_);
	cudaFree(tokenBuffer_);
	cudaFree(attentionMask_);
}
__half* SwinBlockLayer::Forward(__half* data){
	const auto* residual1 = data;
	data = norm1_->Forward(data);
	TokensToWindows(data, windowedInput_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	data = attention_->Forward(windowedInput_);
	WindowsToTokens(data, tokenBuffer_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	data = attnDrop_->Forward(tokenBuffer_);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixFwd_, outDesc_, residual1, &mixFwd_, outDesc_, data));
	const auto* residual2 = data;
	data = norm2_->Forward(data);
	data = fc1_->Forward(data);
	data = gelu_->Forward(data);
	data = fc2_->Forward(data);
	data = ffDrop_->Forward(data);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixFwd_, outDesc_, residual2, &mixFwd_, outDesc_, data));
	return data;
}
__half* SwinBlockLayer::Backward(__half* grad){
	const auto* residual2 = grad;
	grad = ffDrop_->Backward(grad);
	grad = fc2_->Backward(grad);
	grad = gelu_->Backward(grad);
	grad = fc1_->Backward(grad);
	grad = norm2_->Backward(grad);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixBwd_, outDesc_, residual2, &mixBwd_, outDesc_, grad));
	const auto* residual1 = grad;
	grad = attnDrop_->Backward(grad);
	TokensToWindows(grad, windowedGrad_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	grad = attention_->Backward(windowedGrad_);
	WindowsToTokens(grad, tokenBuffer_, batchSize_, tokens_, embedDim_, patchRows_, patchCols_, windowHeight_, windowWidth_, shiftHeight_, shiftWidth_);
	grad = norm1_->Backward(tokenBuffer_);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &mixBwd_, outDesc_, residual1, &mixBwd_, outDesc_, grad));
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
