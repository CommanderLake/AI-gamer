#include "LSTMLayer.h"
#include "common.h"
#include <iostream>
LSTMLayer::LSTMLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int seqLength, int inC, int outC, int numLayers, const char* layerName, bool train) : cudnnHandle_(cudnnHandle),
	cublasHandle_(cublasHandle), batchSize_(batchSize), seqLength_(seqLength), inC_(inC), outC_(outC), numLayers_(numLayers){
	layerName_ = layerName;
	train_ = train;
	checkCUDNN(cudnnCreateRNNDescriptor(&rnnDesc_));
	checkCUDNN(cudnnSetRNNDescriptor_v6(cudnnHandle_, rnnDesc_, outC_, numLayers_, nullptr, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL, CUDNN_LSTM, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_HALF));
	inDesc_ = new cudnnTensorDescriptor_t[seqLength_];
	outDescs_ = new cudnnTensorDescriptor_t[seqLength_];
	for(int i = 0; i < seqLength_; ++i){
		checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_[i]));
		checkCUDNN(cudnnCreateTensorDescriptor(&outDescs_[i]));
		checkCUDNN(cudnnSetTensor4dDescriptor(inDesc_[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, 1, 1));
		checkCUDNN(cudnnSetTensor4dDescriptor(outDescs_[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, outC_, 1, 1));
	}
	outDesc_ = outDescs_[0];
	checkCUDNN(cudnnCreateTensorDescriptor(&hxDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&cxDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&hyDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&cyDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(hxDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, numLayers_, batchSize_, outC_, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(cxDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, numLayers_, batchSize_, outC_, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(hyDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, numLayers_, batchSize_, outC_, 1));
	checkCUDNN(cudnnSetTensor4dDescriptor(cyDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, numLayers_, batchSize_, outC_, 1));
	checkCUDNN(cudnnCreateFilterDescriptor(&wDesc_));
	size_t sizeInBytes;
	checkCUDNN(cudnnGetRNNParamsSize(cudnnHandle_, rnnDesc_, inDesc_[0], &sizeInBytes, CUDNN_DATA_HALF));
	weightCount_ = sizeInBytes/sizeof(__half);
	checkCUDNN(cudnnSetFilter4dDescriptor(wDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, 1, weightCount_, 1, 1));
	checkCUDA(cudaMalloc(&weights_, weightCount_*sizeof(__half)));
	checkCUDA(cudaMalloc(&gradWeights_, weightCount_*sizeof(__half)));
	checkCUDA(cudaMalloc(&hx_, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&cx_, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&hy_, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&cy_, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&inData_, batchSize_*inC_*sizeof(__half)));
	checkCUDA(cudaMalloc(&outData_, batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMemset(weights_, 0, weightCount_*sizeof(__half)));
	checkCUDA(cudaMemset(gradWeights_, 0, weightCount_*sizeof(__half)));
	checkCUDA(cudaMemset(hx_, 0, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMemset(cx_, 0, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMemset(hy_, 0, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMemset(cy_, 0, numLayers_*batchSize_*outC_*sizeof(__half)));
	checkCUDA(cudaMemset(inData_, 0, batchSize_*inC_*sizeof(__half)));
	checkCUDA(cudaMemset(outData_, 0, batchSize_*outC_*sizeof(__half)));
	HeInit(weights_, weightCount_, inC_);
	checkCUDNN(cudnnGetRNNWorkspaceSize(cudnnHandle_, rnnDesc_, seqLength_, inDesc_, &workspaceSize_));
	checkCUDA(cudaMalloc(&workspace_, workspaceSize_));
	checkCUDA(cudaMemset(workspace_, 0, workspaceSize_));
	checkCUDNN(cudnnGetRNNTrainingReserveSize(cudnnHandle_, rnnDesc_, seqLength_, inDesc_, &reserveSpaceSize_));
	checkCUDA(cudaMalloc(&reserveSpace_, reserveSpaceSize_));
	checkCUDA(cudaMemset(reserveSpace_, 0, reserveSpaceSize_));
	if(useAdamW_){
		checkCUDA(cudaMalloc(&m_Weights_, weightCount_*sizeof(float)));
		checkCUDA(cudaMalloc(&v_Weights_, weightCount_*sizeof(float)));
		checkCUDA(cudaMemset(m_Weights_, 0, weightCount_*sizeof(float)));
		checkCUDA(cudaMemset(v_Weights_, 0, weightCount_*sizeof(float)));
	}
}
LSTMLayer::~LSTMLayer(){
	cudaFree(weights_);
	cudaFree(gradWeights_);
	cudaFree(hx_);
	cudaFree(cx_);
	cudaFree(hy_);
	cudaFree(cy_);
	cudaFree(inData_);
	cudaFree(outData_);
	cudaFree(workspace_);
	cudaFree(reserveSpace_);
	for(int i = 0; i < seqLength_; ++i){
		checkCUDNN(cudnnDestroyTensorDescriptor(inDesc_[i]));
		checkCUDNN(cudnnDestroyTensorDescriptor(outDescs_[i]));
	}
	delete[] inDesc_;
	delete[] outDescs_;
	checkCUDNN(cudnnDestroyRNNDescriptor(rnnDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(hxDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(cxDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(hyDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(cyDesc_));
	checkCUDNN(cudnnDestroyFilterDescriptor(wDesc_));
}
__half* LSTMLayer::Forward(__half* data, bool train){
	inData_ = data;
	if(train){
		checkCUDNN(
			cudnnRNNForwardTraining(cudnnHandle_, rnnDesc_, seqLength_, inDesc_, data, hxDesc_, hx_, cxDesc_, cx_, wDesc_, weights_, outDescs_, outData_, hyDesc_, hy_, cyDesc_, cy_, workspace_, workspaceSize_, reserveSpace_,
				reserveSpaceSize_));
	} else{ checkCUDNN(cudnnRNNForwardInference(cudnnHandle_, rnnDesc_, seqLength_, inDesc_, data, hxDesc_, hx_, cxDesc_, cx_, wDesc_, weights_, outDescs_, outData_, hyDesc_, hy_, cyDesc_, cy_, workspace_, workspaceSize_)); }
	//PrintDataHalf(outData_, outC_, "");
	return outData_;
}
__half* LSTMLayer::Backward(__half* grad){
	checkCUDNN(
		cudnnRNNBackwardData(cudnnHandle_, rnnDesc_, seqLength_, outDescs_, outData_, outDescs_, grad, hyDesc_, hy_, cyDesc_, cy_, wDesc_, weights_, hxDesc_, hx_, cxDesc_, cx_, inDesc_, inData_, hxDesc_, gradWeights_, cxDesc_,
			gradWeights_, workspace_, workspaceSize_, reserveSpace_, reserveSpaceSize_));
	checkCUDNN(cudnnRNNBackwardWeights(cudnnHandle_, rnnDesc_, seqLength_, inDesc_, inData_, hxDesc_, hx_, outDescs_, outData_, workspace_, workspaceSize_, wDesc_, gradWeights_, reserveSpace_, reserveSpaceSize_));
	return gradWeights_;
}
void LSTMLayer::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWHalf(weights_, m_Weights_, v_Weights_, learningRate, gradWeights_, weightCount_, t_, 0.0001F);
		++t_;
	} else{
		SGDHalf(weights_, learningRate, gradWeights_, weightCount_);
	}
}
void LSTMLayer::SaveParameters(std::ofstream& file, float* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
}
void LSTMLayer::LoadParameters(std::ifstream& file, float* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
}
void LSTMLayer::SaveOptimizerState(std::ofstream& file, float* buffer){
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(float));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(float), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(float));
}
void LSTMLayer::LoadOptimizerState(std::ifstream& file, float* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(float));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(float), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(float));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(float), cudaMemcpyHostToDevice);
}
bool LSTMLayer::HasParameters(){ return true; }
bool LSTMLayer::HasOptimizerState(){ return useAdamW_; }
size_t LSTMLayer::GetParameterSize(){ return weightCount_*sizeof(__half); }
size_t LSTMLayer::GetOptimizerStateSize(){ return weightCount_*sizeof(float); }