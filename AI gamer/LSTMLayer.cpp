#include "LSTMLayer.h"
#include "common.h"
#include <iostream>
LSTMLayer::LSTMLayer(cudnnHandle_t cudnnHandle, int seqLength, int numLayers, int hiddenSize, int batchSize, int inC, const char* layerName, bool train) : cudnnHandle_(cudnnHandle), inData_(nullptr), batchSize_(batchSize), seqLength_(seqLength),
																																							hiddenSize_(hiddenSize), inC_(inC), numLayers_(numLayers){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = seqLength_*batchSize_*inC_;
	checkCUDNN(cudnnDropoutGetStatesSize(cudnnHandle_, &stateSize_));
	CUDAMallocZero(&dropoutStates_, stateSize_);
	checkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc_));
	checkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc_, cudnnHandle_, 0.5f, dropoutStates_, stateSize_, static_cast<unsigned long long>(time(nullptr))));
	checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc_));
	checkCUDNN(cudnnSetRNNDescriptor(cudnnHandle_, rnnDesc_, hiddenSize_, numLayers_, dropoutDesc_, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_LSTM, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_HALF));
	checkCUDNN(cudnnRNNSetClip(cudnnHandle_, rnnDesc_, CUDNN_RNN_CLIP_MINMAX, CUDNN_PROPAGATE_NAN, -5.0, 5.0));
	checkCUDNN(cudnnSetRNNMatrixMathType(rnnDesc_, CUDNN_TENSOR_OP_MATH));//S
	xDesc_ = new cudnnTensorDescriptor_t[seqLength_];
	yDesc_ = new cudnnTensorDescriptor_t[seqLength_];
	for(int i = 0; i<seqLength_; ++i){
		checkCUDNN(cudnnCreateTensorDescriptor(&xDesc_[i]));
		checkCUDNN(cudnnCreateTensorDescriptor(&yDesc_[i]));
		checkCUDNN(cudnnSetTensor4dDescriptor(xDesc_[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, 1, 1));
		checkCUDNN(cudnnSetTensor4dDescriptor(yDesc_[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, hiddenSize_, 1, 1));
	}
	checkCUDNN(cudnnCreateTensorDescriptor(&hDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&cDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(hDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, numLayers_, batchSize_, hiddenSize_, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(cDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, numLayers_, batchSize_, hiddenSize_, 1));
	checkCUDNN(cudnnGetRNNParamsSize(cudnnHandle_, rnnDesc_, xDesc_[0], &weightSpaceSize_, CUDNN_DATA_HALF));
	weightCount_ = weightSpaceSize_/sizeof(__half);
	checkCUDNN(cudnnCreateFilterDescriptor(&weightDesc_));
	checkCUDNN(cudnnSetFilter4dDescriptor(weightDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, 1, weightCount_, 1, 1));
	CUDAMallocZero(&weights_, weightSpaceSize_);
	CUDAMallocZero(&outData_, seqLength_*batchSize_*hiddenSize_*sizeof(__half));
	HeInit(weights_, weightCount_, inC_+hiddenSize_);
	CUDAMallocZero(&dx_, seqLength_*batchSize_*hiddenSize_*sizeof(__half));
	checkCUDNN(cudnnGetRNNWorkspaceSize(cudnnHandle_, rnnDesc_, seqLength_, xDesc_, &workspaceSize_));
	CUDAMallocZero(&workspace_, workspaceSize_);
	checkCUDNN(cudnnGetRNNTrainingReserveSize(cudnnHandle_, rnnDesc_, seqLength_, xDesc_, &reserveSpaceSize_));
	CUDAMallocZero(&reserveSpace_, reserveSpaceSize_);
	if(train_){
		CUDAMallocZero(&gradWeights_, weightSpaceSize_);
		CUDAMallocZero(&gradOut_, seqLength_*batchSize_*inC_*sizeof(__half));
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightSpaceSize_);
			CUDAMallocZero(&v_Weights_, weightSpaceSize_);
		}
	}
}
LSTMLayer::~LSTMLayer(){
	cudaFree(weights_);
	cudaFree(outData_);
	cudaFree(dx_);
	cudaFree(workspace_);
	cudaFree(reserveSpace_);
	checkCUDNN(cudnnDestroyRNNDescriptor(rnnDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(hDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(cDesc_));
	checkCUDNN(cudnnDestroyFilterDescriptor(weightDesc_));
	for(int i = 0; i<seqLength_; ++i){
		checkCUDNN(cudnnDestroyTensorDescriptor(xDesc_[i]));
		checkCUDNN(cudnnDestroyTensorDescriptor(yDesc_[i]));
	}
	delete[] xDesc_;
	delete[] yDesc_;
	if(train_){
		cudaFree(gradOut_);
		cudaFree(gradWeights_);
		if(useAdamW_){
			cudaFree(m_Weights_);
			cudaFree(v_Weights_);
		}
	}
}
__half* LSTMLayer::Forward(__half* data){
	inData_ = data;
	if(train_){
		checkCUDNN(cudnnRNNForwardTraining(
			cudnnHandle_, rnnDesc_, seqLength_,
			xDesc_, data,
			nullptr, nullptr,
			nullptr, nullptr,
			weightDesc_, weights_,
			yDesc_, outData_,
			nullptr, nullptr,
			nullptr, nullptr,
			workspace_, workspaceSize_,
			reserveSpace_, reserveSpaceSize_
		));
	} else{
		checkCUDNN(cudnnRNNForwardInference(
			cudnnHandle_, rnnDesc_, seqLength_,
			xDesc_, data,
			nullptr, nullptr,
			nullptr, nullptr,
			weightDesc_, weights_,
			yDesc_, outData_,
			nullptr, nullptr,
			nullptr, nullptr,
			workspace_, workspaceSize_
		));
	}
	return outData_ + (seqLength_ - 1)*batchSize_*hiddenSize_;
}
__half* LSTMLayer::Backward(__half* grad){
	checkCUDA(cudaMemcpy(dx_ + (seqLength_ - 1)*batchSize_*hiddenSize_, grad, batchSize_*hiddenSize_*sizeof(__half), cudaMemcpyDeviceToDevice));
	checkCUDNN(cudnnRNNBackwardData(
		cudnnHandle_, rnnDesc_, seqLength_,
		yDesc_, outData_,
		yDesc_, dx_,
		nullptr, nullptr,
		nullptr, nullptr,
		weightDesc_, weights_,
		nullptr, nullptr,
		nullptr, nullptr,
		xDesc_, gradOut_,
		nullptr, nullptr,
		nullptr, nullptr,
		workspace_, workspaceSize_,
		reserveSpace_, reserveSpaceSize_
	));
	checkCUDNN(cudnnRNNBackwardWeights(
		cudnnHandle_, rnnDesc_, seqLength_,
		xDesc_, inData_,
		nullptr, nullptr,
		yDesc_, outData_,
		workspace_, workspaceSize_,
		weightDesc_, gradWeights_,
		reserveSpace_, reserveSpaceSize_
	));
	return gradOut_;
}
void LSTMLayer::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, weightCount_);
		++t_;
	} else{ SGDHalf(weights_, gradWeights_, weightCount_, learningRate, weightDecay_); }
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