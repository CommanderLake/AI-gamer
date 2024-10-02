#include "LSTMLayerv8.h"
#include "common.h"
#include <iostream>
LSTMLayer::LSTMLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int seqLength, int inCHW, int outC, int numLayers, const char* layerName, bool train) : cudnnHandle_(cudnnHandle),
	cublasHandle_(cublasHandle), batchSize_(batchSize), seqLength_(seqLength), inCHW_(inCHW), outC_(outC), numLayers_(numLayers){
	layerName_ = layerName;
	train_ = train;
	checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc_));
	checkCUDNN(cudnnSetRNNDescriptor_v8( rnnDesc_, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_SINGLE_INP_BIAS, CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, CUDNN_DATA_HALF, CUDNN_DATA_HALF, CUDNN_TENSOR_OP_MATH, inCHW_, outC_, outC_, numLayers_, nullptr, 0 ));
	checkCUDNN(cudnnCreateRNNDataDescriptor(&xDesc_));
	checkCUDNN(cudnnCreateRNNDataDescriptor(&yDesc_));
	constexpr cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
	const auto seqLengthArray = new int[batchSize_];
	for(int i = 0; i<batchSize_; ++i){ seqLengthArray[i] = seqLength_; }
	checkCUDA(cudaMalloc(&seqLengthArray_, batchSize_*sizeof(int)));
	checkCUDA(cudaMemcpy(seqLengthArray_, seqLengthArray, batchSize_*sizeof(int), cudaMemcpyHostToDevice));
	checkCUDNN(cudnnSetRNNDataDescriptor( xDesc_, CUDNN_DATA_HALF, layout, seqLength_, batchSize_, inCHW_, seqLengthArray, nullptr ));
	checkCUDNN(cudnnSetRNNDataDescriptor( yDesc_, CUDNN_DATA_HALF, layout, seqLength_, batchSize_, inCHW_, seqLengthArray, nullptr ));
	checkCUDNN(cudnnCreateTensorDescriptor(&hDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&cDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor( hDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, numLayers_, batchSize_, outC_, 1 ));
	checkCUDNN(cudnnSetTensor4dDescriptor( cDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, numLayers_, batchSize_, outC_, 1 ));
	checkCUDNN(cudnnGetRNNWeightSpaceSize(cudnnHandle_, rnnDesc_, &weightSpaceSize_));
	weightCount_ = weightSpaceSize_/sizeof(__half);
	checkCUDA(cudaMalloc(&weights_, weightSpaceSize_));
	checkCUDA(cudaMalloc(&hy_, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&cy_, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&outData_, batchSize_*inCHW_*sizeof(__half)));
	checkCUDA(cudaMemset(weights_, 0, weightSpaceSize_));
	checkCUDA(cudaMemset(hy_, 0, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMemset(cy_, 0, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMemset(outData_, 0, batchSize_*inCHW_*sizeof(__half)));
	HeInit(weights_, weightCount_, inCHW_);
	fwdMode_ = train ? CUDNN_FWD_MODE_TRAINING : CUDNN_FWD_MODE_INFERENCE;
	checkCUDNN(cudnnGetRNNTempSpaceSizes( cudnnHandle_, rnnDesc_, fwdMode_, xDesc_, &workspaceSize_, &reserveSpaceSize_ ));
	checkCUDA(cudaMalloc(&workspace_, workspaceSize_));
	checkCUDA(cudaMemset(workspace_, 0, workspaceSize_));
	checkCUDA(cudaMalloc(&reserveSpace_, reserveSpaceSize_));
	checkCUDA(cudaMemset(reserveSpace_, 0, reserveSpaceSize_));
	if(train_){
		checkCUDA(cudaMalloc(&gradWeights_, weightSpaceSize_));
		checkCUDA(cudaMalloc(&gradOut_, batchSize_*inCHW_*sizeof(__half)));
		checkCUDA(cudaMemset(gradWeights_, 0, weightSpaceSize_));
		checkCUDA(cudaMemset(gradOut_, 0, batchSize_*inCHW_*sizeof(__half)));
		if(useAdamW_){
			checkCUDA(cudaMalloc(&m_Weights_, weightSpaceSize_*sizeof(__half)));
			checkCUDA(cudaMalloc(&v_Weights_, weightSpaceSize_*sizeof(__half)));
			checkCUDA(cudaMemset(m_Weights_, 0, weightSpaceSize_*sizeof(__half)));
			checkCUDA(cudaMemset(v_Weights_, 0, weightSpaceSize_*sizeof(__half)));
		}
	}
	cudnnCreateTensorDescriptor(&outDesc_);
	cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, 1, 1);
}
LSTMLayer::~LSTMLayer(){
	cudaFree(weights_);
	cudaFree(hy_);
	cudaFree(cy_);
	cudaFree(outData_);
	cudaFree(workspace_);
	cudaFree(reserveSpace_);
	cudaFree(seqLengthArray_);
	checkCUDNN(cudnnDestroyRNNDescriptor(rnnDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(hDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(cDesc_));
	checkCUDNN(cudnnDestroyRNNDataDescriptor(xDesc_));
	checkCUDNN(cudnnDestroyRNNDataDescriptor(yDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
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
	checkCUDNN(cudnnRNNForward(cudnnHandle_, rnnDesc_, fwdMode_, seqLengthArray_, xDesc_, data, yDesc_, outData_, hDesc_, nullptr, hy_, cDesc_, nullptr, cy_, weightSpaceSize_, weights_, workspaceSize_, workspace_, reserveSpaceSize_, reserveSpace_));
	return hy_;
}
__half* LSTMLayer::Backward(__half* grad){
	checkCUDNN(cudnnRNNBackwardWeights_v8( cudnnHandle_, rnnDesc_, CUDNN_WGRAD_MODE_ADD, seqLengthArray_, xDesc_, inData_, hDesc_, nullptr, yDesc_, outData_, weightSpaceSize_, gradWeights_, workspaceSize_, workspace_, reserveSpaceSize_, reserveSpace_ ));
	checkCUDNN(cudnnRNNBackwardData_v8( cudnnHandle_, rnnDesc_, seqLengthArray_, yDesc_, outData_, grad, xDesc_, gradOut_, hDesc_, nullptr, nullptr, nullptr, cDesc_, nullptr, nullptr, nullptr, weightSpaceSize_, weights_, workspaceSize_, workspace_, reserveSpaceSize_, reserveSpace_ ));
	return gradWeights_;
}
void LSTMLayer::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, 0.0001F, weightCount_);
		++t_;
	} else{ SGDHalf(weights_, gradWeights_, weightCount_, learningRate); }
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