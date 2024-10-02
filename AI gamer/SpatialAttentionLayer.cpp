#include "common.h"
#include "SpatialAttentionLayer.h"
SpatialAttentionLayer::SpatialAttentionLayer(cudnnHandle_t cudnnHandle, int attentionChannels, int batchSize, int channels, int height, int width, const char* layerName, bool train, float weightDecay) : cudnnHandle_(cudnnHandle),
	attC_(attentionChannels), batchSize_(batchSize), inC_(channels), inH_(height), inW_(width), inData_(nullptr), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*inC_*inH_*inW_;
	attNCHW_ = batchSize_*attC_*inH_*inW_;
	checkCUDNN(cudnnCreateTensorDescriptor(&attentionDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(attentionDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, attC_, inH_, inW_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, inC_, inH_, inW_));
	checkCUDNN(cudnnCreateFilterDescriptor(&keyQueryFilterDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&valueFilterDesc_));
	checkCUDNN(cudnnSetFilter4dDescriptor(keyQueryFilterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, inC_, inC_, 3, 3));
	checkCUDNN(cudnnSetFilter4dDescriptor(valueFilterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, attC_, inC_, 1, 1));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&keyQueryConvDesc_));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&valueConvDesc_));
	checkCUDNN(cudnnSetConvolution2dDescriptor(keyQueryConvDesc_, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolution2dDescriptor(valueConvDesc_, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolutionMathType(keyQueryConvDesc_, CUDNN_TENSOR_OP_MATH)); //S
	dwWeightSize_ = inC_*9;
	pwWeightSize_ = inC_*attC_;
	CUDAMallocZero(&keyMap_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&queryMap_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&valueMap_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&attentionScores_, attNCHW_*sizeof(__half));
	CUDAMallocZero(&keyWeights_, dwWeightSize_*sizeof(__half));
	CUDAMallocZero(&queryWeights_, dwWeightSize_*sizeof(__half));
	CUDAMallocZero(&valueWeights_, pwWeightSize_*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	if(train_){
		HeInit(keyWeights_, dwWeightSize_, 9*inC_);
		HeInit(queryWeights_, dwWeightSize_, 9*inC_);
		HeInit(valueWeights_, pwWeightSize_, inC_);
		CUDAMallocZero(&gradKeyWeights_, dwWeightSize_*sizeof(__half));
		CUDAMallocZero(&gradQueryWeights_, dwWeightSize_*sizeof(__half));
		CUDAMallocZero(&gradValueWeights_, pwWeightSize_*sizeof(__half));
		//CUDAMallocZero(&gradKeyMap_, outNCHW_*sizeof(__half));
		//CUDAMallocZero(&gradQueryMap_, outNCHW_*sizeof(__half));
		//CUDAMallocZero(&gradValueMap_, outNCHW_*sizeof(__half));
		//CUDAMallocZero(&gradAttention_, attNCHW_*sizeof(__half));
		CUDAMallocZero(&gradOut_, outNCHW_*sizeof(__half));
		if(useAdamW_){
			CUDAMallocZero(&m_Key_, dwWeightSize_*sizeof(__half));
			CUDAMallocZero(&v_Key_, dwWeightSize_*sizeof(__half));
			CUDAMallocZero(&m_Query_, dwWeightSize_*sizeof(__half));
			CUDAMallocZero(&v_Query_, dwWeightSize_*sizeof(__half));
			CUDAMallocZero(&m_Pointwise_, pwWeightSize_*sizeof(__half));
			CUDAMallocZero(&v_Pointwise_, pwWeightSize_*sizeof(__half));
		}
	}
	keyQueryAlgos_ = GetConvolutionAlgorithms(cudnnHandle_, outDesc_, keyQueryFilterDesc_, keyQueryConvDesc_, outDesc_, train_);
	valueAlgos_ = GetConvolutionAlgorithms(cudnnHandle_, outDesc_, valueFilterDesc_, valueConvDesc_, attentionDesc_, train_);
	CUDAMallocZero(&keyQueryWorkspace_, keyQueryAlgos_.workspaceSize);
	CUDAMallocZero(&valueWorkspace_, valueAlgos_.workspaceSize);
}
SpatialAttentionLayer::~SpatialAttentionLayer(){
	cudaFree(keyMap_);
	cudaFree(queryMap_);
	cudaFree(valueMap_);
	cudaFree(attentionScores_);
	cudaFree(keyWeights_);
	cudaFree(queryWeights_);
	cudaFree(valueWeights_);
	cudaFree(outData_);
	cudaFree(keyQueryWorkspace_);
	cudaFree(valueWorkspace_);
	if(train_){
		cudaFree(gradKeyWeights_);
		cudaFree(gradQueryWeights_);
		cudaFree(gradValueWeights_);
		//cudaFree(gradKeyMap_);
		//cudaFree(gradQueryMap_);
		//cudaFree(gradValueMap_);
		//cudaFree(gradAttention_);
		cudaFree(gradOut_);
		if(useAdamW_){
			cudaFree(m_Key_);
			cudaFree(v_Key_);
			cudaFree(m_Query_);
			cudaFree(v_Query_);
			cudaFree(m_Pointwise_);
			cudaFree(v_Pointwise_);
		}
	}
	cudnnDestroyTensorDescriptor(outDesc_);
	cudnnDestroyTensorDescriptor(attentionDesc_);
	cudnnDestroyFilterDescriptor(keyQueryFilterDesc_);
	cudnnDestroyFilterDescriptor(valueFilterDesc_);
	cudnnDestroyConvolutionDescriptor(keyQueryConvDesc_);
	cudnnDestroyConvolutionDescriptor(valueConvDesc_);
}
__half* SpatialAttentionLayer::Forward(__half* data){
	inData_ = data;
	/*// Depthwise convolution for key and query
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, outDesc_, inData_, keyQueryFilterDesc_, keyWeights_, keyQueryConvDesc_, keyQueryAlgos_.fwdAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta0, outDesc_, keyMap_));
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, outDesc_, inData_, keyQueryFilterDesc_, queryWeights_, keyQueryConvDesc_, keyQueryAlgos_.fwdAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta0, outDesc_, queryMap_));
	// Pointwise convolution to compute value
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, outDesc_, inData_, valueFilterDesc_, valueWeights_, valueConvDesc_, valueAlgos_.fwdAlgo, valueWorkspace_, valueAlgos_.workspaceSize, &beta0, attentionDesc_, valueMap_));
	// Combine key and query information
	ComputeAttention(queryMap_, keyMap_, attentionScores_, inC_, attC_, inH_, inW_, attNCHW_);
	// Apply softmax to attention scores
	checkCUDNN(cudnnSoftmaxForward(cudnnHandle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, attentionDesc_, attentionScores_, &beta0, attentionDesc_, attentionScores_));
	// Apply attention to input
	ApplyAttention(valueMap_, attentionScores_, outData_, inC_, attC_, inH_, inW_, outNCHW_);
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, outDesc_, inData_, &beta1, outDesc_, outData_));*/
	SpatialAttentionForward(inData_, keyWeights_, queryWeights_, valueWeights_, keyMap_, queryMap_, valueMap_, attentionScores_, outData_, batchSize_, inC_, attC_, inH_, inW_);
	return outData_;
}
__half* SpatialAttentionLayer::Backward(__half* grad){
	/*// Gradient of attention application
	ApplyAttentionBackward(grad, valueMap_, attentionScores_, gradValueMap_, gradAttention_, inC_, attC_, inH_, inW_, outNCHW_);
	// Gradient of softmax
	checkCUDNN(cudnnSoftmaxBackward(cudnnHandle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, attentionDesc_, attentionScores_, attentionDesc_, gradAttention_, &beta0, attentionDesc_, gradAttention_));
	// Gradient of combining key and query
	ComputeQueryKeyGrad(gradAttention_, queryMap_, keyMap_, gradQueryMap_, gradKeyMap_, inC_, attC_, inH_, inW_, outNCHW_);
	// Gradient of Value convolution
	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &alpha, outDesc_, keyMap_, attentionDesc_, gradValueMap_, valueConvDesc_, valueAlgos_.bwdFilterAlgo, valueWorkspace_, valueAlgos_.workspaceSize, &beta0, valueFilterDesc_, gradValueWeights_));
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &alpha, valueFilterDesc_, valueWeights_, attentionDesc_, gradValueMap_, valueConvDesc_, valueAlgos_.bwdDataAlgo, valueWorkspace_, valueAlgos_.workspaceSize, &beta0, outDesc_, gradOut_));
	// Gradient of Query convolution
	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &alpha, outDesc_, inData_, outDesc_, gradQueryMap_, keyQueryConvDesc_, keyQueryAlgos_.bwdFilterAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta0, keyQueryFilterDesc_, gradQueryWeights_));
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &alpha, keyQueryFilterDesc_, queryWeights_, outDesc_, gradQueryMap_, keyQueryConvDesc_, keyQueryAlgos_.bwdDataAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta1, outDesc_, gradOut_));
	// Gradient of Key convolution
	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &alpha, outDesc_, inData_, outDesc_, gradKeyMap_, keyQueryConvDesc_, keyQueryAlgos_.bwdFilterAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta0, keyQueryFilterDesc_, gradKeyWeights_));
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &alpha, keyQueryFilterDesc_, keyWeights_, outDesc_, gradKeyMap_, keyQueryConvDesc_, keyQueryAlgos_.bwdDataAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta1, outDesc_, gradOut_));
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, outDesc_, grad, &beta1, outDesc_, gradOut_));*/
	SpatialAttentionBackward(inData_, keyWeights_, queryWeights_, valueWeights_, keyMap_, queryMap_, valueMap_, attentionScores_, grad, gradOut_, gradKeyWeights_, gradQueryWeights_, gradValueWeights_, batchSize_, inC_, attC_, inH_, inW_);
	return gradOut_;
}
void SpatialAttentionLayer::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWHalf(keyWeights_, gradKeyWeights_, m_Key_, v_Key_, learningRate, t_, weightDecay_, dwWeightSize_);
		AdamWHalf(queryWeights_, gradQueryWeights_, m_Query_, v_Query_, learningRate, t_, weightDecay_, dwWeightSize_);
		AdamWHalf(valueWeights_, gradValueWeights_, m_Pointwise_, v_Pointwise_, learningRate, t_, weightDecay_, pwWeightSize_);
		++t_;
	} else{
		SGDHalf(keyWeights_, gradKeyWeights_, dwWeightSize_, learningRate, weightDecay_);
		SGDHalf(queryWeights_, gradQueryWeights_, dwWeightSize_, learningRate, weightDecay_);
		SGDHalf(valueWeights_, gradValueWeights_, pwWeightSize_, learningRate, weightDecay_);
	}
}
void SpatialAttentionLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, keyWeights_, dwWeightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(buffer, queryWeights_, dwWeightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(buffer, valueWeights_, pwWeightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), pwWeightSize_*sizeof(__half));
}
void SpatialAttentionLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(keyWeights_, buffer, dwWeightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(queryWeights_, buffer, dwWeightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), pwWeightSize_*sizeof(__half));
	cudaMemcpy(valueWeights_, buffer, pwWeightSize_*sizeof(__half), cudaMemcpyHostToDevice);
}
void SpatialAttentionLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Key_, dwWeightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(buffer, v_Key_, dwWeightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(buffer, m_Query_, dwWeightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(buffer, v_Query_, dwWeightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(buffer, m_Pointwise_, pwWeightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), pwWeightSize_*sizeof(__half));
	cudaMemcpy(buffer, v_Pointwise_, pwWeightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), pwWeightSize_*sizeof(__half));
}
void SpatialAttentionLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(m_Key_, buffer, dwWeightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(v_Key_, buffer, dwWeightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(m_Query_, buffer, dwWeightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), dwWeightSize_*sizeof(__half));
	cudaMemcpy(v_Query_, buffer, dwWeightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), pwWeightSize_*sizeof(__half));
	cudaMemcpy(m_Pointwise_, buffer, pwWeightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), pwWeightSize_*sizeof(__half));
	cudaMemcpy(v_Pointwise_, buffer, pwWeightSize_*sizeof(__half), cudaMemcpyHostToDevice);
}
size_t SpatialAttentionLayer::GetParameterSize(){
	return dwWeightSize_*sizeof(__half);
}
size_t SpatialAttentionLayer::GetOptimizerStateSize(){
	return dwWeightSize_*sizeof(__half);
}