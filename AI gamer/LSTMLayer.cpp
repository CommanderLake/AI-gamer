#include "LSTMLayer.h"
#include "common.h"
#include "CuCommon.cuh"
#include <iostream>
LSTMLayer::LSTMLayer(const cudnnHandle_t cudnnHandle, const int seqLength, const int numLayers, const int hiddenSize, const int batchSize, const int inC, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength) :
	cudnnHandle_(cudnnHandle), batchSize_(batchSize), seqLength_(seqLength), hiddenSize_(hiddenSize), inC_(inC), numLayers_(numLayers), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*inC_;
	alphaWeights_ = 1.0f/(batchSize_*gradAccumLength_);
	checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc_));
	checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&hcxyDesc_));
	checkCUDNN(cudnnCreateRNNDataDescriptor(&xDesc_));
	checkCUDNN(cudnnCreateRNNDataDescriptor(&yDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&weightDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&wTensDesc_));
	checkCUDNN(cudnnDropoutGetStatesSize(cudnnHandle_, &stateSize_));
	CUDAMallocZero(&dropoutStates_, stateSize_);
	checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc_, cudnnHandle_, 0.2f, dropoutStates_, stateSize_, static_cast<unsigned long long>(time(nullptr))));
	checkCUDNN(cudnnSetRNNDescriptor(cudnnHandle_, rnnDesc_, hiddenSize_, numLayers_, dropoutDesc_, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_LSTM, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_HALF));
	checkCUDNN(cudnnRNNSetClip(cudnnHandle_, rnnDesc_, CUDNN_RNN_CLIP_MINMAX, CUDNN_PROPAGATE_NAN, -5.0, 5.0));
	checkCUDNN(cudnnSetRNNMatrixMathType(rnnDesc_, CUDNN_TENSOR_OP_MATH)); //S
	checkCUDNN(cudnnSetRNNPaddingMode(rnnDesc_, CUDNN_RNN_PADDED_IO_ENABLED));
	xDescs_ = new cudnnTensorDescriptor_t[seqLength_];
	for(int i = 0; i<seqLength_; ++i){
		checkCUDNN(cudnnCreateTensorDescriptor(&xDescs_[i]));
		checkCUDNN(cudnnSetTensor4dDescriptor(xDescs_[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, 1, 1));
	}
	checkCUDNN(cudnnSetTensor4dDescriptor(hcxyDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, numLayers_, batchSize_, hiddenSize_, 1));
	seqLengths_ = new int[batchSize_];
	for(int i = 0; i<batchSize_; ++i){ seqLengths_[i] = seqLength_; }
	checkCUDNN(cudnnSetRNNDataDescriptor(xDesc_, CUDNN_DATA_HALF, CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED, seqLength_, batchSize_, inC_, seqLengths_, nullptr));
	checkCUDNN(cudnnSetRNNDataDescriptor(yDesc_, CUDNN_DATA_HALF, CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED, seqLength_, batchSize_, hiddenSize_, seqLengths_, nullptr));
	checkCUDNN(cudnnGetRNNParamsSize(cudnnHandle_, rnnDesc_, xDescs_[0], &weightSpaceSize_, CUDNN_DATA_HALF));
	weightCount_ = weightSpaceSize_/sizeof(__half);
	checkCUDNN(cudnnSetFilter4dDescriptor(weightDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, 1, weightCount_, 1, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(wTensDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 1, 1, weightCount_));
	CUDAMallocZero(&weights_, weightSpaceSize_);
	CUDAMallocZero(&y_, numLayers_*seqLength_*batchSize_*hiddenSize_*sizeof(__half));
	CUDAMallocZero(&hxy_, numLayers_*batchSize_*hiddenSize_*sizeof(__half));
	CUDAMallocZero(&cxy_, numLayers_*batchSize_*hiddenSize_*sizeof(__half));
	checkCUDNN(cudnnGetRNNWorkspaceSize(cudnnHandle_, rnnDesc_, seqLength_, xDescs_, &workspaceSize_));
	CUDAMallocZero(&workspace_, workspaceSize_);
	checkCUDNN(cudnnGetRNNTrainingReserveSize(cudnnHandle_, rnnDesc_, seqLength_, xDescs_, &reserveSpaceSize_));
	CUDAMallocZero(&reserveSpace_, reserveSpaceSize_);
	if(train_){
		OrthogonalInit(weights_, 4*numLayers_*inC_, hiddenSize_);
		CUDAMallocZero(&dGrads_, weightSpaceSize_);
		CUDAMallocZero(&gradWeights_, weightSpaceSize_);
		CUDAMallocZero(&dx_, seqLength_*batchSize_*inC_*sizeof(__half));
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightSpaceSize_);
			CUDAMallocZero(&v_Weights_, weightSpaceSize_);
		}
	}
}
LSTMLayer::~LSTMLayer(){
	cudaFree(reserveSpace_);
	cudaFree(workspace_);
	cudaFree(y_);
	cudaFree(weights_);
	for(int i = 0; i<seqLength_; ++i){
		cudnnDestroyTensorDescriptor(xDescs_[i]);
	}
	delete[] seqLengths_;
	delete[] xDescs_;
	cudaFree(dropoutStates_);
	cudnnDestroyFilterDescriptor(weightDesc_);
	cudnnDestroyRNNDataDescriptor(yDesc_);
	cudnnDestroyRNNDataDescriptor(xDesc_);
	cudnnDestroyTensorDescriptor(hcxyDesc_);
	cudnnDestroyRNNDescriptor(rnnDesc_);
	cudnnDestroyDropoutDescriptor(dropoutDesc_);
	if(train_){
		cudaFree(dx_);
		cudaFree(gradWeights_);
		cudaFree(dGrads_);
		if(useAdamW_){
			cudaFree(v_Weights_);
			cudaFree(m_Weights_);
		}
	}
}
__half* LSTMLayer::Forward(__half* x){
	x_ = x;
	if(train_){
		checkCUDNN(cudnnRNNForwardTrainingEx(
			cudnnHandle_, rnnDesc_, xDesc_, x_, hcxyDesc_, hxy_, hcxyDesc_, cxy_, weightDesc_, weights_, yDesc_, y_, hcxyDesc_, hxy_, hcxyDesc_, cxy_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, workspace_, workspaceSize_, reserveSpace_, reserveSpaceSize_));
	} else{
		checkCUDNN(cudnnRNNForwardInferenceEx(
			cudnnHandle_, rnnDesc_, xDesc_, x_, hcxyDesc_, hxy_, hcxyDesc_, cxy_, weightDesc_, weights_, yDesc_, y_, hcxyDesc_, hxy_, hcxyDesc_, cxy_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, workspace_, workspaceSize_));
	}
	return y_;
}
__half* LSTMLayer::Backward(__half* dy){
	checkCUDNN(cudnnRNNBackwardDataEx(
		cudnnHandle_, rnnDesc_, yDesc_, y_, yDesc_, dy, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, weightDesc_, weights_, nullptr, nullptr, nullptr, nullptr, xDesc_, dx_, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, workspace_, workspaceSize_,
		reserveSpace_, reserveSpaceSize_));
	checkCUDNN(cudnnRNNBackwardWeightsEx(cudnnHandle_, rnnDesc_, xDesc_, x_, nullptr, nullptr, yDesc_, y_, workspace_, workspaceSize_, weightDesc_, gradWeights_, reserveSpace_, reserveSpaceSize_));
	const float* betaWeights = accumCount_++%gradAccumLength_==0 ? &beta0_ : &beta1_;
	cudnnAddTensor(cudnnHandle_, &alphaWeights_, wTensDesc_, gradWeights_, betaWeights, wTensDesc_, dGrads_);
	return dx_;
}
void LSTMLayer::UpdateParameters(const float learningRate){
	if(accumCount_%gradAccumLength_>0) return;
	if(useAdamW_){
		++t_;
		AdamWHalf(weights_, dGrads_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, weightCount_);
	} else{ SGDHalf(weights_, dGrads_, weightCount_, learningRate, weightDecay_); }
}
void LSTMLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}
void LSTMLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
void LSTMLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}
void LSTMLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
size_t LSTMLayer::GetParameterSize(){ return weightCount_*sizeof(__half); }
size_t LSTMLayer::GetOptimizerStateSize(){ return weightCount_*sizeof(__half); }