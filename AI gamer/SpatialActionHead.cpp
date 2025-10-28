#include "SpatialActionHead.h"
#include "common.h"
#include "CuCommon.cuh"
#include "ConvLayer.h"
#include "BatchNorm.h"
#include "GELULayer.h"
#include "Dropout.h"
#include "FCLayer.h"
#include "AsinhLayer.h"
#include "LayerNorm.h"
#include "TokensToSpatialLayer.h"
#include "ViewerLayer.h"
#undef min
#undef max
SpatialActionHead::SpatialActionHead(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int patchRows, const int patchCols, const int embedSize, const char* layerName, const bool train, const float weightDecay,
	const int gradAccumLength) : cudnn_(cudnnHandle), cublas_(cublasHandle), batchSize_(batchSize), nTokens_(patchRows*patchCols), embedSize_(embedSize), patchRows_(patchRows), patchCols_(patchCols),
	sharedHeight_(patchRows), sharedWidth_(patchCols), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*NUM_CTRLS_;
	tokensWithCls_ = nTokens_ + 1;
	CUDAMallocZero(&predictions_, batchSize_*NUM_CTRLS_*sizeof(__half));
	CUDAMallocZero(&classTokens_, static_cast<size_t>(batchSize_)*embedSize_*sizeof(__half));
	CUDAMallocZero(&patchTokens_, static_cast<size_t>(batchSize_)*nTokens_*embedSize_*sizeof(__half));
	CUDAMallocZero(&upstreamGrad_, static_cast<size_t>(batchSize_)*tokensWithCls_*embedSize_*sizeof(__half));
	int sharedH = patchRows_;
	int sharedW = patchCols_;
	trunkC1_ = RoundUp(embedSize_/4, 16);
	//sharedLayers_.push_back(new ViewerLayer(tokenElems, embedDim_, patchRows_, patchCols_, 16, "TokensToSpatial Viewer", true, 1.0f, false));
	sharedLayers_.push_back(new TokensToSpatialLayer(batchSize_, nTokens_, embedSize_, patchRows_, patchCols_, "TokensToSpatialLayer", train_));
	sharedLayers_.push_back(new ConvLayer(cudnn_, batchSize_, embedSize_, trunkC1_, 1, 1, 0, &sharedH, &sharedW, 1, "Spatial Trunk Conv 1", train_, weightDecay_, gradAccumLength_, Xavier));
	sharedLayers_.push_back(new BatchNorm(cudnn_, CUDNN_BATCHNORM_SPATIAL, batchSize_, trunkC1_, sharedH, sharedW, "Trunk BN 1", train_, gradAccumLength_));
	sharedLayers_.push_back(new GELULayer(batchSize_, trunkC1_, sharedH, sharedW, "Spatial Trunk GELU 1"));
	sharedLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, trunkC1_, sharedH, sharedW, "Trunk Drop 1", train_));
	sharedHeight_ = sharedH;
	sharedWidth_ = sharedW;
	const int sharedSpatialSize = sharedHeight_*sharedWidth_;
	sharedOutC_ = 4096;
	checkCUDNN(cudnnCreateTensorDescriptor(&neckDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(neckDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, sharedOutC_, 1, 1));
	const size_t classElems = static_cast<size_t>(batchSize_)*sharedOutC_;
	CUDAMallocZero(&classSharedWorkspace_, classElems*2*sizeof(__half));
	classNeckGrad_ = classSharedWorkspace_;
	sharedGradBuffer_ = classSharedWorkspace_ + classElems;
	sharedLayers_.push_back(new FCLayer(cublas_, batchSize_, trunkC1_*sharedSpatialSize, sharedOutC_, "Trunk-neck FC 1", train_, weightDecay_, gradAccumLength_, Xavier, true));
	sharedLayers_.push_back(new LayerNorm(batchSize_, sharedOutC_, 1, 1, "Trunk-neck LN", train_));
	sharedLayers_.push_back(new GELULayer(batchSize_, sharedOutC_, 1, 1, "Trunk-neck GELU"));
	sharedLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, sharedOutC_, 1, 1, "Trunk-neck Drop", train_));
	classLayers_.push_back(new FCLayer(cublas_, batchSize_, trunkC1_, sharedOutC_, "CLS FC", train_, weightDecay_, gradAccumLength_, Xavier, true));
	classLayers_.push_back(new LayerNorm(batchSize_, sharedOutC_, 1, 1, "CLS LN", train_));
	classLayers_.push_back(new GELULayer(batchSize_, sharedOutC_, 1, 1, "CLS GELU"));
	classLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, sharedOutC_, 1, 1, "CLS Drop", train_));
	headC_ = 1024;
	const size_t headBatchElems = static_cast<size_t>(batchSize_)*headC_;
	const size_t weightElems = static_cast<size_t>(batchSize_)*headC_*sharedOutC_;
	CUDAMallocZero(&hyperOutWorkspace_, headBatchElems*2*sizeof(__half));
	buttonHyperOut_ = hyperOutWorkspace_;
	axisHyperOut_ = hyperOutWorkspace_ + headBatchElems;
	CUDAMallocZero(&weightGradWorkspace_, weightElems*2*sizeof(__half));
	buttonWeightGrad_ = weightGradWorkspace_;
	axisWeightGrad_ = weightGradWorkspace_ + weightElems;
	const int weightOutDim = headC_*sharedOutC_;
	hyperLayers_.push_back(new FCLayer(cublas_, batchSize_, sharedOutC_, weightOutDim, "CLS->Buttons Weight", train_, weightDecay_, gradAccumLength_, Xavier, false));
	hyperLayers_.push_back(new FCLayer(cublas_, batchSize_, sharedOutC_, headC_, "CLS->Buttons Bias", train_, weightDecay_, gradAccumLength_, Xavier, false));
	hyperLayers_.push_back(new FCLayer(cublas_, batchSize_, sharedOutC_, weightOutDim, "CLS->Axes Weight", train_, weightDecay_, gradAccumLength_, Xavier, false));
	hyperLayers_.push_back(new FCLayer(cublas_, batchSize_, sharedOutC_, headC_, "CLS->Axes Bias", train_, weightDecay_, gradAccumLength_, Xavier, false));
	buttonLayers_.push_back(new LayerNorm(batchSize_, headC_, 1, 1, "Buttons LN", train_));
	buttonLayers_.push_back(new GELULayer(batchSize_, headC_, 1, 1, "Buttons GELU"));
	buttonLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, headC_, 1, 1, "Buttons Drop", train_));
	buttonLayers_.push_back(new FCLayer(cublas_, batchSize_, headC_, NUM_BUTS_, "Buttons FC", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new LayerNorm(batchSize_, headC_, 1, 1, "Axes LN", train_));
	axisLayers_.push_back(new GELULayer(batchSize_, headC_, 1, 1, "Axes GELU"));
	axisLayers_.push_back(new Dropout(cudnn_, 0.1f, batchSize_, headC_, 1, 1, "Axes Drop", train_));
	axisLayers_.push_back(new FCLayer(cublas_, batchSize_, headC_, NUM_AXES_, "Axes FC", train_, weightDecay_, gradAccumLength_, Xavier, true));
	axisLayers_.push_back(new AsinhLayer(batchSize_, NUM_AXES_, 1, 1, static_cast<int>(AXIS_SCALE_), "Axes Asinh"));
	checkCUDNN(cudnnCreateTensorDescriptor(&headDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(headDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, headC_, 1, 1));
}
SpatialActionHead::~SpatialActionHead(){
	cudaFree(predictions_);
	cudaFree(upstreamGrad_);
	cudaFree(classSharedWorkspace_);
	classSharedWorkspace_ = nullptr;
	classNeckGrad_ = nullptr;
	sharedGradBuffer_ = nullptr;
	cudaFree(patchTokens_);
	cudaFree(classTokens_);
	cudaFree(hyperOutWorkspace_);
	hyperOutWorkspace_ = nullptr;
	buttonHyperOut_ = nullptr;
	axisHyperOut_ = nullptr;
	cudaFree(weightGradWorkspace_);
	weightGradWorkspace_ = nullptr;
	buttonWeightGrad_ = nullptr;
	axisWeightGrad_ = nullptr;
	hyperOutputs_.fill(nullptr);
	cudnnDestroyTensorDescriptor(neckDesc_);
	cudnnDestroyTensorDescriptor(headDesc_);
	for(const auto* layer : hyperLayers_) delete layer;
	for(const auto* layer : classLayers_) delete layer;
	for(const auto* layer : axisLayers_) delete layer;
	for(const auto* layer : buttonLayers_) delete layer;
	for(const auto* layer : sharedLayers_) delete layer;
	hyperLayers_.clear();
	classLayers_.clear();
	axisLayers_.clear();
	buttonLayers_.clear();
	sharedLayers_.clear();
}
__half* SpatialActionHead::Forward(__half* data){
	StripClassToken(data, patchTokens_, batchSize_, embedSize_, nTokens_);
	auto tokenInput = patchTokens_;
	for(auto* layer : sharedLayers_){ tokenInput = layer->Forward(tokenInput); }
	sharedOutput_ = tokenInput;
	GatherClassTokens(data, classTokens_, batchSize_, tokensWithCls_, embedSize_);
	auto classData = classTokens_;
	for(auto* layer : classLayers_){ classData = layer->Forward(classData); }
	for(int i = 0; i < kNumHyperLayers; ++i){
		hyperOutputs_[i] = hyperLayers_[i]->Forward(classData);
	}
	buttonWeightParams_ = hyperOutputs_[kButtonWeightIndex];
	const auto buttonBias = hyperOutputs_[kButtonBiasIndex];
	axisWeightParams_ = hyperOutputs_[kAxisWeightIndex];
	const auto axisBias = hyperOutputs_[kAxisBiasIndex];
	const long long weightStride = static_cast<long long>(headC_)*sharedOutC_;
	const long long inputStride = sharedOutC_;
	const long long outputStride = headC_;
	checkCUBLAS(cublasGemmStridedBatchedEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, headC_, 1, sharedOutC_, &one_, buttonWeightParams_, CUDA_R_16F, headC_, weightStride, sharedOutput_, CUDA_R_16F, sharedOutC_, inputStride, &zero_, buttonHyperOut_, CUDA_R_16F, headC_, outputStride, batchSize_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmStridedBatchedEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, headC_, 1, sharedOutC_, &one_, axisWeightParams_, CUDA_R_16F, headC_, weightStride, sharedOutput_, CUDA_R_16F, sharedOutC_, inputStride, &zero_, axisHyperOut_, CUDA_R_16F, headC_, outputStride, batchSize_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUDNN(cudnnAddTensor(cudnn_, &one_, headDesc_, buttonBias, &one_, headDesc_, buttonHyperOut_));
	checkCUDNN(cudnnAddTensor(cudnn_, &one_, headDesc_, axisBias, &one_, headDesc_, axisHyperOut_));
	auto buttonData = buttonHyperOut_;
	for(auto* layer : buttonLayers_){ buttonData = layer->Forward(buttonData); }
	auto axisData = axisHyperOut_;
	for(auto* layer : axisLayers_){ axisData = layer->Forward(axisData); }
	MergeOutputs(predictions_, buttonData, axisData, NUM_CTRLS_, NUM_BUTS_, NUM_CTRLS_*batchSize_);
	return predictions_;
}
__half* SpatialActionHead::Backward(__half* grad){
	auto buttonGrad = grad;
	auto axisGrad = grad + NUM_BUTS_*batchSize_;
	for(int i = static_cast<int>(buttonLayers_.size()); --i >= 0;){ buttonGrad = buttonLayers_[i]->Backward(buttonGrad); }
	for(int i = static_cast<int>(axisLayers_.size()); --i >= 0;){ axisGrad = axisLayers_[i]->Backward(axisGrad); }
	const long long weightStride = static_cast<long long>(headC_)*sharedOutC_;
	const long long gradStride = headC_;
	const long long sharedStride = sharedOutC_;
	const int classElemCount = batchSize_*sharedOutC_;
	checkCUDA(cudaMemset(sharedGradBuffer_, 0, static_cast<size_t>(classElemCount)*sizeof(__half)));
	checkCUBLAS(cublasGemmStridedBatchedEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, sharedOutC_, 1, headC_, &one_, buttonWeightParams_, CUDA_R_16F, headC_, weightStride, buttonGrad, CUDA_R_16F, headC_, gradStride, &zero_, sharedGradBuffer_, CUDA_R_16F, sharedOutC_, sharedStride, batchSize_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmStridedBatchedEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, sharedOutC_, 1, headC_, &one_, axisWeightParams_, CUDA_R_16F, headC_, weightStride, axisGrad, CUDA_R_16F, headC_, gradStride, &one_, sharedGradBuffer_, CUDA_R_16F, sharedOutC_, sharedStride, batchSize_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmStridedBatchedEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_T, headC_, sharedOutC_, 1, &one_, buttonGrad, CUDA_R_16F, headC_, gradStride, sharedOutput_, CUDA_R_16F, sharedOutC_, sharedStride, &zero_, buttonWeightGrad_, CUDA_R_16F, headC_, weightStride, batchSize_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUBLAS(cublasGemmStridedBatchedEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_T, headC_, sharedOutC_, 1, &one_, axisGrad, CUDA_R_16F, headC_, gradStride, sharedOutput_, CUDA_R_16F, sharedOutC_, sharedStride, &zero_, axisWeightGrad_, CUDA_R_16F, headC_, weightStride, batchSize_, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
	checkCUDA(cudaMemset(classNeckGrad_, 0, static_cast<size_t>(classElemCount)*sizeof(__half)));
	const std::array<__half*, kNumHyperLayers> hyperBackInputs = {buttonWeightGrad_, buttonGrad, axisWeightGrad_, axisGrad};
	for(int i = 0; i < kNumHyperLayers; ++i){
		const auto classGrad = hyperLayers_[i]->Backward(hyperBackInputs[i]);
		checkCUDNN(cudnnAddTensor(cudnn_, &one_, neckDesc_, classGrad, &one_, neckDesc_, classNeckGrad_));
	}
	auto classBackGrad = classNeckGrad_;
	for(int i = static_cast<int>(classLayers_.size()); --i >= 0;){ classBackGrad = classLayers_[i]->Backward(classBackGrad); }
	auto sharedGrad = sharedGradBuffer_;
	for(int i = static_cast<int>(sharedLayers_.size()); --i >= 0;){ sharedGrad = sharedLayers_[i]->Backward(sharedGrad); }
	BuildClassTokenOutput(classBackGrad, sharedGrad, upstreamGrad_, batchSize_, embedSize_, nTokens_);
	return upstreamGrad_;
}
void SpatialActionHead::UpdateParameters(const float learningRate){
	for(auto* layer : sharedLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : buttonLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : axisLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : hyperLayers_){ layer->UpdateParameters(learningRate); }
	for(auto* layer : classLayers_){ layer->UpdateParameters(learningRate); }
}
void SpatialActionHead::SaveParameters(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : hyperLayers_){ layer->SaveParameters(file, buffer); }
	for(auto* layer : classLayers_){ layer->SaveParameters(file, buffer); }
}
void SpatialActionHead::LoadParameters(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : hyperLayers_){ layer->LoadParameters(file, buffer); }
	for(auto* layer : classLayers_){ layer->LoadParameters(file, buffer); }
}
void SpatialActionHead::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : hyperLayers_){ layer->SaveOptimizerState(file, buffer); }
	for(auto* layer : classLayers_){ layer->SaveOptimizerState(file, buffer); }
}
void SpatialActionHead::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	for(auto* layer : sharedLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : buttonLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : axisLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : hyperLayers_){ layer->LoadOptimizerState(file, buffer); }
	for(auto* layer : classLayers_){ layer->LoadOptimizerState(file, buffer); }
}
size_t SpatialActionHead::GetParameterSize(){
	size_t maxSize = 0;
	for(auto* layer : sharedLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : hyperLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	for(auto* layer : classLayers_){ maxSize = std::max(maxSize, layer->GetParameterSize()); }
	return maxSize;
}
size_t SpatialActionHead::GetOptimizerStateSize(){
	size_t maxSize = 0;
	for(auto* layer : sharedLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : buttonLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : axisLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : hyperLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	for(auto* layer : classLayers_){ maxSize = std::max(maxSize, layer->GetOptimizerStateSize()); }
	return maxSize;
}
void SpatialActionHead::SetTrain(const bool enable){
	train_ = enable;
	for(auto* layer : sharedLayers_){ layer->SetTrain(enable); }
	for(auto* layer : buttonLayers_){ layer->SetTrain(enable); }
	for(auto* layer : axisLayers_){ layer->SetTrain(enable); }
	for(auto* layer : hyperLayers_){ layer->SetTrain(enable); }
	for(auto* layer : classLayers_){ layer->SetTrain(enable); }
}