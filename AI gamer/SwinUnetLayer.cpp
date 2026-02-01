#include "SwinUnetLayer.h"
#include "CuCommon.cuh"
#include "LayerNorm.h"
#include "PatchEmbedLayer.h"
#include "PatchExpandingLayer.h"
#include "PatchMergingLayer.h"
#include "SwinBlockLayer.h"
#include <algorithm>
#include <stdexcept>

#include "GELULayer.h"
SwinUnetLayer::SwinUnetLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int inChannels, const int inHeight, const int inWidth, const int patchSize, const int embedH, const int embedW, const int blocksPerStage, const int numStages, const int baseHeads, const int baseWindowSize, const float maxDropPathRate, std::string layerName, const bool train, const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), batchSize_(batchSize), inChannels_(inChannels), inHeight_(inHeight), inWidth_(inWidth), patchSize_(patchSize), embedH_(embedH), embedW_(embedW), blocksPerStage_(blocksPerStage), numStages_(numStages), baseHeads_(baseHeads), baseWindowSize_(baseWindowSize), maxDropPathRate_(maxDropPathRate), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength), weightInitMethod_(weightInitMethod){
	layerName_ = layerName;
	train_ = train;
	const int embedSize = embedH_ * embedW_;
	const int patchRows = DivCeil(inHeight_, patchSize_);
	const int patchCols = DivCeil(inWidth_, patchSize_);
	{
		size_t maxWindowElems = 0;
		size_t maxTokenElems = 0;
		auto updateWorkspaceSizes = [&](const int tokens, const int embedDim, const int patchRowsSize, const int patchColsSize, const int windowHeight, const int windowWidth){
			const int windowTokens = windowHeight * windowWidth;
			const int windowCount = (patchRowsSize / windowHeight) * (patchColsSize / windowWidth);
			const int windowBatch = batchSize_ * windowCount;
			const size_t windowElems = static_cast<size_t>(windowBatch) * windowTokens * embedDim;
			const size_t tokenElems = static_cast<size_t>(batchSize_) * tokens * embedDim;
			maxWindowElems = std::max(maxWindowElems, windowElems);
			maxTokenElems = std::max(maxTokenElems, tokenElems);
		};
		int nTokensWorkspace = patchRows * patchCols;
		int embedDimWorkspace = embedSize;
		int currentPatchRows = patchRows;
		int currentPatchCols = patchCols;
		for(int stage = 0; stage < numStages_; ++stage){
			const int windowHeight = std::min(baseWindowSize_, currentPatchRows);
			const int windowWidth = std::min(baseWindowSize_, currentPatchCols);
			updateWorkspaceSizes(nTokensWorkspace, embedDimWorkspace, currentPatchRows, currentPatchCols, windowHeight, windowWidth);
			currentPatchRows /= 2;
			currentPatchCols /= 2;
			nTokensWorkspace = currentPatchRows * currentPatchCols;
			embedDimWorkspace *= 2;
		}
		for(int stage = 0; stage < numStages_; ++stage){
			currentPatchRows *= 2;
			currentPatchCols *= 2;
			nTokensWorkspace = currentPatchRows * currentPatchCols;
			embedDimWorkspace /= 2;
			const int windowHeight = std::min(baseWindowSize_, currentPatchRows);
			const int windowWidth = std::min(baseWindowSize_, currentPatchCols);
			updateWorkspaceSizes(nTokensWorkspace, embedDimWorkspace, currentPatchRows, currentPatchCols, windowHeight, windowWidth);
		}
		blockWorkspace_.windowBytes = maxWindowElems * sizeof(__half);
		blockWorkspace_.tokenBytes = maxTokenElems * sizeof(__half);
		CUDAMallocZero(&blockWorkspace_.windowedInput, blockWorkspace_.windowBytes);
		CUDAMallocZero(&blockWorkspace_.windowedGrad, blockWorkspace_.windowBytes);
		CUDAMallocZero(&blockWorkspace_.tokens, blockWorkspace_.tokenBytes);
		CUDAMallocZero(&blockWorkspace_.residualGrad, blockWorkspace_.tokenBytes);
	}
	patchEmbed_ = new PatchEmbedLayer(cudnnHandle_, cublasHandle_, batchSize_, inChannels_, inHeight_, inWidth_, patchSize_, embedSize, "PatchEmbed", train_, weightDecay_, gradAccumLength_, weightInitMethod_);
	int nTokens = patchRows * patchCols;
	int embedDim = embedSize;
	int ffDim = embedDim * 4;
	int currentPatchRows = patchRows;
	int currentPatchCols = patchCols;
	int blockIndex = 0;
	const int totalBlocks = numStages_ * blocksPerStage_ * 2;
	encoderStages_.reserve(numStages_);
	for(int stage = 0; stage < numStages_; ++stage){
		EncoderStage encoderStage;
		const int stageHeads = baseHeads_ << stage;
		const int windowHeight = std::min(baseWindowSize_, currentPatchRows);
		const int windowWidth = std::min(baseWindowSize_, currentPatchCols);
		const int shiftHeight = windowHeight > 1 ? windowHeight / 2 : 0;
		const int shiftWidth = windowWidth > 1 ? windowWidth / 2 : 0;
		encoderStage.blocks.reserve(blocksPerStage_);
		for(int block = 0; block < blocksPerStage_; ++block){
			const float dropPathRate = totalBlocks > 1 ? maxDropPathRate_ * (static_cast<float>(blockIndex) / static_cast<float>(totalBlocks - 1)) : 0.0f;
			auto name = "SwinBlock" + std::to_string(blockIndex);
			const bool useShift = block % 2 != 0;
			const int blockShiftHeight = useShift ? shiftHeight : 0;
			const int blockShiftWidth = useShift ? shiftWidth : 0;
			encoderStage.blocks.push_back(new SwinBlockLayer(cudnnHandle_, cublasHandle_, batchSize_, nTokens, embedDim, ffDim, stageHeads, currentPatchRows, currentPatchCols, windowHeight, windowWidth, blockShiftHeight, blockShiftWidth, dropPathRate, name.c_str(), train_, weightDecay_, gradAccumLength_, weightInitMethod_, blockWorkspace_.windowedInput, blockWorkspace_.windowedGrad, blockWorkspace_.tokens, blockWorkspace_.residualGrad));
			++blockIndex;
		}
		encoderStage.skip.elements = batchSize_ * nTokens * embedDim;
		encoderStage.skip.bytes = static_cast<size_t>(encoderStage.skip.elements) * sizeof(__half);
		CUDAMallocZero(&encoderStage.skip.data, encoderStage.skip.bytes);
		CUDAMallocZero(&encoderStage.skip.grad, encoderStage.skip.bytes);
		auto mergeName = "PatchMerge" + std::to_string(stage);
		encoderStage.merge = new PatchMergingLayer(cudnnHandle_, cublasHandle_, batchSize_, nTokens, embedDim, currentPatchRows, currentPatchCols, mergeName.c_str(), train_, weightDecay_, gradAccumLength_, weightInitMethod_);
		encoderStages_.push_back(encoderStage);
		currentPatchRows /= 2;
		currentPatchCols /= 2;
		nTokens = currentPatchRows * currentPatchCols;
		embedDim *= 2;
		ffDim = embedDim * 4;
	}
	decoderStages_.reserve(numStages_);
	for(int stage = 0; stage < numStages_; ++stage){
		DecoderStage decoderStage;
		auto expandName = "PatchExpand" + std::to_string(stage);
		decoderStage.expand = new PatchExpandingLayer(cudnnHandle_, cublasHandle_, batchSize_, nTokens, embedDim, currentPatchRows, currentPatchCols, expandName.c_str(), train_, weightDecay_, gradAccumLength_, weightInitMethod_);
		currentPatchRows *= 2;
		currentPatchCols *= 2;
		nTokens = currentPatchRows * currentPatchCols;
		embedDim /= 2;
		ffDim = embedDim * 4;
		const int stageHeads = baseHeads_ << (numStages_ - stage - 1);
		const int windowHeight = std::min(baseWindowSize_, currentPatchRows);
		const int windowWidth = std::min(baseWindowSize_, currentPatchCols);
		const int shiftHeight = windowHeight > 1 ? windowHeight / 2 : 0;
		const int shiftWidth = windowWidth > 1 ? windowWidth / 2 : 0;
		decoderStage.blocks.reserve(blocksPerStage_);
		for(int block = 0; block < blocksPerStage_; ++block){
			const float dropPathRate = totalBlocks > 1 ? maxDropPathRate_ * (static_cast<float>(blockIndex) / static_cast<float>(totalBlocks - 1)) : 0.0f;
			auto name = "SwinBlockUp" + std::to_string(blockIndex);
			const bool useShift = block % 2 != 0;
			const int blockShiftHeight = useShift ? shiftHeight : 0;
			const int blockShiftWidth = useShift ? shiftWidth : 0;
			decoderStage.blocks.push_back(new SwinBlockLayer(cudnnHandle_, cublasHandle_, batchSize_, nTokens, embedDim, ffDim, stageHeads, currentPatchRows, currentPatchCols, windowHeight, windowWidth, blockShiftHeight, blockShiftWidth, dropPathRate, name.c_str(), train_, weightDecay_, gradAccumLength_, weightInitMethod_, blockWorkspace_.windowedInput, blockWorkspace_.windowedGrad, blockWorkspace_.tokens, blockWorkspace_.residualGrad));
			++blockIndex;
		}
		decoderStages_.push_back(decoderStage);
	}
	postNorm_ = new LayerNorm(batchSize_ * nTokens, embedDim, 1, 1, "Post-encoder norm", train_);
	postGELU_ = new GELULayer(batchSize_*nTokens, embedDim, 1, 1, "Post-encoder GELU");
	outNCHW_ = static_cast<size_t>(batchSize_) * nTokens * embedDim;
}
SwinUnetLayer::~SwinUnetLayer(){
	delete patchEmbed_;
	delete postNorm_;
	delete postGELU_;
	for(auto& stage : encoderStages_){
		for(const auto* block : stage.blocks) delete block;
		delete stage.merge;
		cudaFree(stage.skip.data);
		cudaFree(stage.skip.grad);
	}
	for(auto& stage : decoderStages_){
		delete stage.expand;
		for(const auto* block : stage.blocks) delete block;
	}
	cudaFree(blockWorkspace_.windowedInput);
	cudaFree(blockWorkspace_.windowedGrad);
	cudaFree(blockWorkspace_.tokens);
	cudaFree(blockWorkspace_.residualGrad);
}
__half* SwinUnetLayer::Forward(__half* data){
	data = patchEmbed_->Forward(data);
	for(size_t stage = 0; stage < encoderStages_.size(); ++stage){
		auto& encoderStage = encoderStages_[stage];
		for(auto* block : encoderStage.blocks){ data = block->Forward(data); }
		checkCUDA(cudaMemcpy(encoderStage.skip.data, data, encoderStage.skip.bytes, cudaMemcpyDeviceToDevice));
		data = encoderStage.merge->Forward(data);
	}
	for(size_t stage = 0; stage < decoderStages_.size(); ++stage){
		auto& decoderStage = decoderStages_[stage];
		data = decoderStage.expand->Forward(data);
		const size_t skipIndex = encoderStages_.size() - stage - 1;
		const auto& skip = encoderStages_[skipIndex].skip;
		AddTensor(1.0f, data, 1.0f, skip.data, skip.elements);
		for(auto* block : decoderStage.blocks){ data = block->Forward(data); }
	}
	data = postNorm_->Forward(data);
	data = postGELU_->Forward(data);
	return data;
}
__half* SwinUnetLayer::Backward(__half* grad){
	for(const auto& stage : encoderStages_){ checkCUDA(cudaMemset(stage.skip.grad, 0, stage.skip.bytes)); }
	grad = postGELU_->Backward(grad);
	grad = postNorm_->Backward(grad);
	for(size_t stage = decoderStages_.size(); stage-- > 0;){
		auto& decoderStage = decoderStages_[stage];
		for(size_t i = decoderStage.blocks.size(); i-- > 0;){ grad = decoderStage.blocks[i]->Backward(grad); }
		const size_t skipIndex = encoderStages_.size() - stage - 1;
		const auto& skip = encoderStages_[skipIndex].skip;
		AddTensor(0.0f, skip.grad, 1.0f, grad, skip.elements);
		grad = decoderStage.expand->Backward(grad);
	}
	for(size_t stage = encoderStages_.size(); stage-- > 0;){
		auto& encoderStage = encoderStages_[stage];
		grad = encoderStage.merge->Backward(grad);
		AddTensor(1.0f, grad, 1.0f, encoderStage.skip.grad, encoderStage.skip.elements);
		for(size_t i = encoderStage.blocks.size(); i-- > 0;){ grad = encoderStage.blocks[i]->Backward(grad); }
	}
	return patchEmbed_->Backward(grad);
}
void SwinUnetLayer::UpdateParameters(const float lr){
	patchEmbed_->UpdateParameters(lr);
	for(auto& stage : encoderStages_){
		for(auto* block : stage.blocks){ block->UpdateParameters(lr); }
		stage.merge->UpdateParameters(lr);
	}
	for(auto& stage : decoderStages_){
		stage.expand->UpdateParameters(lr);
		for(auto* block : stage.blocks){ block->UpdateParameters(lr); }
	}
	postNorm_->UpdateParameters(lr);
}
void SwinUnetLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	patchEmbed_->SaveParameters(file, buffer);
	for(auto& stage : encoderStages_){
		for(auto* block : stage.blocks){ block->SaveParameters(file, buffer); }
		stage.merge->SaveParameters(file, buffer);
	}
	for(auto& stage : decoderStages_){
		stage.expand->SaveParameters(file, buffer);
		for(auto* block : stage.blocks){ block->SaveParameters(file, buffer); }
	}
	postNorm_->SaveParameters(file, buffer);
}
void SwinUnetLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	patchEmbed_->LoadParameters(file, buffer);
	for(auto& stage : encoderStages_){
		for(auto* block : stage.blocks){ block->LoadParameters(file, buffer); }
		stage.merge->LoadParameters(file, buffer);
	}
	for(auto& stage : decoderStages_){
		stage.expand->LoadParameters(file, buffer);
		for(auto* block : stage.blocks){ block->LoadParameters(file, buffer); }
	}
	postNorm_->LoadParameters(file, buffer);
}
void SwinUnetLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	patchEmbed_->SaveOptimizerState(file, buffer);
	for(auto& stage : encoderStages_){
		for(auto* block : stage.blocks){ block->SaveOptimizerState(file, buffer); }
		stage.merge->SaveOptimizerState(file, buffer);
	}
	for(auto& stage : decoderStages_){
		stage.expand->SaveOptimizerState(file, buffer);
		for(auto* block : stage.blocks){ block->SaveOptimizerState(file, buffer); }
	}
	postNorm_->SaveOptimizerState(file, buffer);
}
void SwinUnetLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	patchEmbed_->LoadOptimizerState(file, buffer);
	for(auto& stage : encoderStages_){
		for(auto* block : stage.blocks){ block->LoadOptimizerState(file, buffer); }
		stage.merge->LoadOptimizerState(file, buffer);
	}
	for(auto& stage : decoderStages_){
		stage.expand->LoadOptimizerState(file, buffer);
		for(auto* block : stage.blocks){ block->LoadOptimizerState(file, buffer); }
	}
	postNorm_->LoadOptimizerState(file, buffer);
}
size_t SwinUnetLayer::GetParameterSize(){
	size_t maxSize = patchEmbed_->GetParameterSize();
	for(auto& stage : encoderStages_){
		for(auto* block : stage.blocks){ maxSize = std::max(maxSize, block->GetParameterSize()); }
		maxSize = std::max(maxSize, stage.merge->GetParameterSize());
	}
	for(auto& stage : decoderStages_){
		maxSize = std::max(maxSize, stage.expand->GetParameterSize());
		for(auto* block : stage.blocks){ maxSize = std::max(maxSize, block->GetParameterSize()); }
	}
	maxSize = std::max(maxSize, postNorm_->GetParameterSize());
	return maxSize;
}
size_t SwinUnetLayer::GetOptimizerStateSize(){
	size_t maxSize = patchEmbed_->GetOptimizerStateSize();
	for(auto& stage : encoderStages_){
		for(auto* block : stage.blocks){ maxSize = std::max(maxSize, block->GetOptimizerStateSize()); }
		maxSize = std::max(maxSize, stage.merge->GetOptimizerStateSize());
	}
	for(auto& stage : decoderStages_){
		maxSize = std::max(maxSize, stage.expand->GetOptimizerStateSize());
		for(auto* block : stage.blocks){ maxSize = std::max(maxSize, block->GetOptimizerStateSize()); }
	}
	maxSize = std::max(maxSize, postNorm_->GetOptimizerStateSize());
	return maxSize;
}
void SwinUnetLayer::SetTrain(const bool enable){
	train_ = enable;
	patchEmbed_->SetTrain(enable);
	for(auto& stage : encoderStages_){
		for(auto* block : stage.blocks){ block->SetTrain(enable); }
		stage.merge->SetTrain(enable);
	}
	for(auto& stage : decoderStages_){
		stage.expand->SetTrain(enable);
		for(auto* block : stage.blocks){ block->SetTrain(enable); }
	}
	postNorm_->SetTrain(enable);
	postGELU_->SetTrain(enable);
}