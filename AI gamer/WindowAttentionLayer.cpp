#include "WindowAttentionLayer.h"
#include "CuCommon.cuh"
#include "WmmaAttentionLayer.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace{
	constexpr float kMaskValue = -100.0f;
	void BuildShiftWindowMask(int height, int width, int windowSize, int shiftSize, std::vector<float>& outMask){
		const int numWindowsH = height / windowSize;
		const int numWindowsW = width / windowSize;
		const int numWindows = numWindowsH * numWindowsW;
		const int windowTokens = windowSize * windowSize;
		std::vector<int> regionIds(height * width, 0);
		int region = 0;
		for(int y = 0; y < height; y += windowSize){
			for(int x = 0; x < width; x += windowSize){
				for(int wy = 0; wy < windowSize; ++wy){
					for(int wx = 0; wx < windowSize; ++wx){
						regionIds[(y + wy) * width + (x + wx)] = region;
					}
				}
				++region;
			}
		}
		std::vector<int> shiftedRegion(height * width, 0);
		for(int y = 0; y < height; ++y){
			for(int x = 0; x < width; ++x){
				const int srcY = (y + shiftSize + height) % height;
				const int srcX = (x + shiftSize + width) % width;
				shiftedRegion[y * width + x] = regionIds[srcY * width + srcX];
			}
		}
		outMask.assign(numWindows * windowTokens * windowTokens, 0.0f);
		for(int winY = 0; winY < numWindowsH; ++winY){
			for(int winX = 0; winX < numWindowsW; ++winX){
				const int windowIndex = winY * numWindowsW + winX;
				std::vector<int> windowLabels(windowTokens, 0);
				for(int wy = 0; wy < windowSize; ++wy){
					for(int wx = 0; wx < windowSize; ++wx){
						const int local = wy * windowSize + wx;
						const int srcY = winY * windowSize + wy;
						const int srcX = winX * windowSize + wx;
						windowLabels[local] = shiftedRegion[srcY * width + srcX];
					}
				}
				for(int i = 0; i < windowTokens; ++i){
					for(int j = 0; j < windowTokens; ++j){
						const float val = windowLabels[i] == windowLabels[j] ? 0.0f : kMaskValue;
						outMask[(windowIndex * windowTokens + i) * windowTokens + j] = val;
					}
				}
			}
		}
	}
}

WindowAttentionLayer::WindowAttentionLayer(const cudnnHandle_t cudnnHandle, const cublasHandle_t cublasHandle, const int batchSize, const int tokens, const int embedDim, const int numHeads, const int patchRows, const int patchCols, const int windowSize, const int shiftSize, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength, const WeightInitMethod weightInitMethod) : cudnnHandle_(cudnnHandle), cublasHandle_(cublasHandle), batchSize_(batchSize), tokens_(tokens), embedDim_(embedDim), numHeads_(numHeads), patchRows_(patchRows), patchCols_(patchCols), windowSize_(windowSize), shiftSize_(shiftSize), gradAccumLength_(gradAccumLength), weightDecay_(weightDecay){
	if(windowSize_ <= 0){
		throw std::invalid_argument("Window size must be positive");
	}
	if(shiftSize_ < 0 || shiftSize_ >= windowSize_){
		throw std::invalid_argument("Shift size must be within [0, windowSize)");
	}
	if(patchRows_ <= 0 || patchCols_ <= 0 || patchRows_*patchCols_ != tokens_){
		throw std::invalid_argument("Token count must match patchRows*patchCols");
	}
	if(patchRows_ % windowSize_ != 0 || patchCols_ % windowSize_ != 0){
		throw std::invalid_argument("Patch grid must be divisible by window size");
	}
	layerName_ = layerName;
	train_ = train;
	windowTokens_ = windowSize_ * windowSize_;
	numWindows_ = (patchRows_ / windowSize_) * (patchCols_ / windowSize_);
	windowBatch_ = batchSize_ * numWindows_;
	const int accumLen = gradAccumLength_ > 0 ? gradAccumLength_ : 1;
	alphaBias_ = 1.0f / (static_cast<float>(windowBatch_) * accumLen);
	relPosBiasSize_ = (2 * windowSize_ - 1) * (2 * windowSize_ - 1);
	outNCHW_ = static_cast<size_t>(batchSize_) * tokens_ * embedDim_;
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_ * tokens_, embedDim_, 1, 1));
	const size_t tokenElems = static_cast<size_t>(batchSize_) * tokens_ * embedDim_;
	const size_t windowElems = static_cast<size_t>(windowBatch_) * windowTokens_ * embedDim_;
	CUDAMallocZero(&shiftedData_, tokenElems * sizeof(__half));
	CUDAMallocZero(&windowData_, windowElems * sizeof(__half));
	CUDAMallocZero(&mergedData_, tokenElems * sizeof(__half));
	CUDAMallocZero(&outData_, tokenElems * sizeof(__half));
	if(train_){
		CUDAMallocZero(&shiftedGrad_, tokenElems * sizeof(__half));
		CUDAMallocZero(&windowGrad_, windowElems * sizeof(__half));
		CUDAMallocZero(&outGrad_, tokenElems * sizeof(__half));
	}
	if(shiftSize_ > 0){
		std::vector<float> mask;
		BuildShiftWindowMask(patchRows_, patchCols_, windowSize_, shiftSize_, mask);
		const size_t maskElems = static_cast<size_t>(windowBatch_) * windowTokens_ * windowTokens_;
		std::vector<float> expanded(maskElems, 0.0f);
		const size_t windowMaskStride = static_cast<size_t>(windowTokens_) * windowTokens_;
		for(int b = 0; b < batchSize_; ++b){
			for(int w = 0; w < numWindows_; ++w){
				const size_t srcOffset = static_cast<size_t>(w) * windowMaskStride;
				const size_t dstOffset = static_cast<size_t>(b * numWindows_ + w) * windowMaskStride;
				std::copy(mask.begin() + srcOffset, mask.begin() + srcOffset + windowMaskStride, expanded.begin() + dstOffset);
			}
		}
		checkCUDA(cudaMalloc(reinterpret_cast<void**>(&attnMask_), maskElems * sizeof(float)));
		checkCUDA(cudaMemcpy(attnMask_, expanded.data(), maskElems * sizeof(float), cudaMemcpyHostToDevice));
	}
	{
		std::vector<int> relPosIndex(windowTokens_ * windowTokens_, 0);
		for(int y1 = 0; y1 < windowSize_; ++y1){
			for(int x1 = 0; x1 < windowSize_; ++x1){
				const int idx1 = y1 * windowSize_ + x1;
				for(int y2 = 0; y2 < windowSize_; ++y2){
					for(int x2 = 0; x2 < windowSize_; ++x2){
						const int idx2 = y2 * windowSize_ + x2;
						const int relY = y1 - y2 + windowSize_ - 1;
						const int relX = x1 - x2 + windowSize_ - 1;
						const int relIndex = relY * (2 * windowSize_ - 1) + relX;
						relPosIndex[idx1 * windowTokens_ + idx2] = relIndex;
					}
				}
			}
		}
		checkCUDA(cudaMalloc(reinterpret_cast<void**>(&relPosIndex_), relPosIndex.size() * sizeof(int)));
		checkCUDA(cudaMemcpy(relPosIndex_, relPosIndex.data(), relPosIndex.size() * sizeof(int), cudaMemcpyHostToDevice));
	}
	CUDAMallocZero(&relPosBias_, static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float));
	if(train_){
		CUDAMallocZero(&gradRelPosBias_, static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float));
		CUDAMallocZero(&mRelPosBias_, static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float));
		CUDAMallocZero(&vRelPosBias_, static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float));
	}
	attention_ = new WmmaAttentionLayer(cudnnHandle_, cublasHandle_, windowBatch_, windowTokens_, embedDim_, numHeads_, "WindowAttentionBase", train_, weightDecay_, gradAccumLength_, weightInitMethod);
	attention_->SetAttentionMask(attnMask_);
	attention_->SetRelativePositionBias(relPosBias_, relPosIndex_, relPosBiasSize_);
}

WindowAttentionLayer::~WindowAttentionLayer(){
	delete attention_;
	cudaFree(attnMask_);
	cudaFree(relPosBias_);
	cudaFree(gradRelPosBias_);
	cudaFree(mRelPosBias_);
	cudaFree(vRelPosBias_);
	cudaFree(relPosIndex_);
	cudaFree(shiftedData_);
	cudaFree(windowData_);
	cudaFree(mergedData_);
	cudaFree(outData_);
	if(train_){
		cudaFree(shiftedGrad_);
		cudaFree(windowGrad_);
		cudaFree(outGrad_);
	}
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
}

__half* WindowAttentionLayer::Forward(__half* data){
	if(shiftSize_ > 0){
		ShiftWindowPartition(data, windowData_, batchSize_, patchRows_, patchCols_, embedDim_, windowSize_, shiftSize_, shiftSize_);
	} else{
		WindowPartition(data, windowData_, batchSize_, patchRows_, patchCols_, embedDim_, windowSize_);
	}
	const auto* windowOut = attention_->Forward(windowData_);
	if(shiftSize_ > 0){
		WindowReverseShift(windowOut, outData_, batchSize_, patchRows_, patchCols_, embedDim_, windowSize_, -shiftSize_, -shiftSize_);
		return outData_;
	}
	WindowReverse(windowOut, mergedData_, batchSize_, patchRows_, patchCols_, embedDim_, windowSize_);
	return mergedData_;
}

__half* WindowAttentionLayer::Backward(__half* grad){
	if(shiftSize_ > 0){
		ShiftWindowPartition(grad, windowGrad_, batchSize_, patchRows_, patchCols_, embedDim_, windowSize_, shiftSize_, shiftSize_);
	} else{
		WindowPartition(grad, windowGrad_, batchSize_, patchRows_, patchCols_, embedDim_, windowSize_);
	}
	const auto* windowInputGrad = attention_->Backward(windowGrad_);
	if(train_ && gradRelPosBias_){
		const int accumLen = gradAccumLength_ > 0 ? gradAccumLength_ : 1;
		if(biasAccumCount_++ % accumLen == 0){
			cudaMemset(gradRelPosBias_, 0, static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float));
		}
		attention_->AccumulateRelPosBiasGrad(gradRelPosBias_, relPosIndex_, relPosBiasSize_, alphaBias_);
	}
	if(shiftSize_ > 0){
		WindowReverseShift(windowInputGrad, outGrad_, batchSize_, patchRows_, patchCols_, embedDim_, windowSize_, -shiftSize_, -shiftSize_);
		return outGrad_;
	}
	WindowReverse(windowInputGrad, mergedData_, batchSize_, patchRows_, patchCols_, embedDim_, windowSize_);
	return mergedData_;
}

void WindowAttentionLayer::UpdateParameters(const float lr){
	attention_->UpdateParameters(lr);
	if(!train_ || !gradRelPosBias_) return;
	const int accumLen = gradAccumLength_ > 0 ? gradAccumLength_ : 1;
	if(biasAccumCount_ % accumLen > 0) return;
	AdamWFloat(relPosBias_, gradRelPosBias_, mRelPosBias_, vRelPosBias_, lr, biasAccumCount_, weightDecay_, numHeads_ * relPosBiasSize_);
}

void WindowAttentionLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	attention_->SaveParameters(file, buffer);
	const size_t biasBytes = static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float);
	cudaMemcpy(buffer, relPosBias_, biasBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), biasBytes);
}

void WindowAttentionLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	attention_->LoadParameters(file, buffer);
	const size_t biasBytes = static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float);
	file.read(reinterpret_cast<char*>(buffer), biasBytes);
	cudaMemcpy(relPosBias_, buffer, biasBytes, cudaMemcpyHostToDevice);
}

void WindowAttentionLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	attention_->SaveOptimizerState(file, buffer);
	if(!train_) return;
	const size_t biasBytes = static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float);
	cudaMemcpy(buffer, mRelPosBias_, biasBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), biasBytes);
	cudaMemcpy(buffer, vRelPosBias_, biasBytes, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<char*>(buffer), biasBytes);
	file.write(reinterpret_cast<char*>(&biasAccumCount_), sizeof(int));
}

void WindowAttentionLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	attention_->LoadOptimizerState(file, buffer);
	if(!train_) return;
	const size_t biasBytes = static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float);
	file.read(reinterpret_cast<char*>(buffer), biasBytes);
	cudaMemcpy(mRelPosBias_, buffer, biasBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), biasBytes);
	cudaMemcpy(vRelPosBias_, buffer, biasBytes, cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(&biasAccumCount_), sizeof(int));
}

size_t WindowAttentionLayer::GetParameterSize(){
	const size_t biasBytes = static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float);
	return std::max(attention_->GetParameterSize(), biasBytes);
}

size_t WindowAttentionLayer::GetOptimizerStateSize(){
	if(!train_) return attention_->GetOptimizerStateSize();
	const size_t biasBytes = static_cast<size_t>(numHeads_) * relPosBiasSize_ * sizeof(float);
	return std::max(attention_->GetOptimizerStateSize(), biasBytes * 2 + sizeof(int));
}

void WindowAttentionLayer::SetTrain(const bool enable){
	train_ = enable;
	attention_->SetTrain(enable);
}
