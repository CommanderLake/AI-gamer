#include "common.h"
#include "SpatialAttentionLayer.h"
SpatialAttentionLayer::SpatialAttentionLayer(cudnnHandle_t cudnnHandle, int attentionChannels, int numHeads, int batchSize, int channels, int height, int width, const char* layerName, bool train, float weightDecay) : cudnnHandle_(cudnnHandle),
	batchSize_(batchSize), inC_(channels), inH_(height), inW_(width), numHeads_(numHeads), attC_(attentionChannels), inData_(nullptr), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	outNCHW_ = batchSize_*inC_*inH_*inW_;
	channelsPerHead_ = inC_/numHeads_;
	attCPerHead_ = attC_/numHeads_;
	checkCUDNN(cudnnCreateTensorDescriptor(&attentionDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(attentionDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, attCPerHead_, inH_, inW_));
	checkCUDNN(cudnnSetTensor4dDescriptor(outDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, batchSize_, channelsPerHead_, inH_, inW_));
	checkCUDNN(cudnnCreateFilterDescriptor(&keyQueryFilterDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&valueFilterDesc_));
	checkCUDNN(cudnnSetFilter4dDescriptor(keyQueryFilterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, channelsPerHead_, channelsPerHead_, 3, 3));
	checkCUDNN(cudnnSetFilter4dDescriptor(valueFilterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, attCPerHead_, channelsPerHead_, 1, 1));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&keyQueryConvDesc_));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&valueConvDesc_));
	checkCUDNN(cudnnSetConvolution2dDescriptor(keyQueryConvDesc_, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolution2dDescriptor(valueConvDesc_, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolutionMathType(keyQueryConvDesc_, CUDNN_TENSOR_OP_MATH)); 
	kqHeadSize_ = batchSize_*channelsPerHead_*inH_*inW_;
	vHeadSize_ = batchSize_*attCPerHead_*inH_*inW_;
	dwWeightSizeSingle_ = channelsPerHead_*channelsPerHead_*9;
	pwWeightSizeSingle_ = channelsPerHead_*attCPerHead_;
	dwWeightSizeAll_ = numHeads_*dwWeightSizeSingle_;
	pwWeightSizeAll_ = numHeads_*channelsPerHead_*attCPerHead_;
	CUDAMallocZero(&keyMap_, kqHeadSize_*numHeads_*sizeof(__half));
	CUDAMallocZero(&queryMap_, kqHeadSize_*numHeads_*sizeof(__half));
	CUDAMallocZero(&valueMap_, vHeadSize_*numHeads_*sizeof(__half));
	CUDAMallocZero(&attentionScores_, vHeadSize_*numHeads_*sizeof(__half));
	CUDAMallocZero(&keyWeights_, dwWeightSizeAll_*sizeof(__half));
	CUDAMallocZero(&queryWeights_, dwWeightSizeAll_*sizeof(__half));
	CUDAMallocZero(&valueWeights_, pwWeightSizeAll_*sizeof(__half));
	CUDAMallocZero(&headOutput_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	if(train_){
		HeInit(keyWeights_, dwWeightSizeAll_, 9*channelsPerHead_);
		HeInit(queryWeights_, dwWeightSizeAll_, 9*channelsPerHead_);
		HeInit(valueWeights_, pwWeightSizeAll_, channelsPerHead_);
		CUDAMallocZero(&gradKeyWeights_, dwWeightSizeAll_*sizeof(__half));
		CUDAMallocZero(&gradQueryWeights_, dwWeightSizeAll_*sizeof(__half));
		CUDAMallocZero(&gradValueWeights_, pwWeightSizeAll_*sizeof(__half));
		CUDAMallocZero(&gradKeyMap_, kqHeadSize_*numHeads_*sizeof(__half));
		CUDAMallocZero(&gradQueryMap_, kqHeadSize_*numHeads_*sizeof(__half));
		CUDAMallocZero(&gradValueMap_, vHeadSize_*numHeads_*sizeof(__half));
		CUDAMallocZero(&gradAttention_, vHeadSize_*numHeads_*sizeof(__half));
		CUDAMallocZero(&gradOut_, outNCHW_*sizeof(__half));
		if(useAdamW_){
			CUDAMallocZero(&m_Key_, dwWeightSizeAll_*sizeof(__half));
			CUDAMallocZero(&v_Key_, dwWeightSizeAll_*sizeof(__half));
			CUDAMallocZero(&m_Query_, dwWeightSizeAll_*sizeof(__half));
			CUDAMallocZero(&v_Query_, dwWeightSizeAll_*sizeof(__half));
			CUDAMallocZero(&m_Pointwise_, pwWeightSizeAll_*sizeof(__half));
			CUDAMallocZero(&v_Pointwise_, pwWeightSizeAll_*sizeof(__half));
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
		cudaFree(gradKeyMap_);
		cudaFree(gradQueryMap_);
		cudaFree(gradValueMap_);
		cudaFree(gradAttention_);
		cudaFree(headOutput_);
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
	const float headAlpha = 1.0f/numHeads_;
	cudaMemset(outData_, 0, outNCHW_*sizeof(__half));
	for(int h = 0; h<numHeads_; h++){
		const size_t inputOffset = h*channelsPerHead_*inH_*inW_;
		const size_t weightOffset = h*dwWeightSizeSingle_;
		const size_t valueWeightOffset = h*pwWeightSizeSingle_;
		const size_t kqOffset = h*kqHeadSize_;
		const size_t vOffset = h*vHeadSize_;
		__half* currentKeyMap = keyMap_ + kqOffset;
		__half* currentQueryMap = queryMap_ + kqOffset;
		__half* currentValueMap = valueMap_ + vOffset;
		__half* currentAttentionScores = attentionScores_ + vOffset;
		const __half* headInput = inData_ + inputOffset;
		checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, outDesc_, headInput, keyQueryFilterDesc_, keyWeights_+weightOffset, keyQueryConvDesc_, keyQueryAlgos_.fwdAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta0, outDesc_, currentKeyMap));
		checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, outDesc_, headInput, keyQueryFilterDesc_, queryWeights_+weightOffset, keyQueryConvDesc_, keyQueryAlgos_.fwdAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta0, outDesc_, currentQueryMap));
		checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, outDesc_, headInput, valueFilterDesc_, valueWeights_+valueWeightOffset, valueConvDesc_, valueAlgos_.fwdAlgo, valueWorkspace_, valueAlgos_.workspaceSize, &beta0, attentionDesc_, currentValueMap));
		ComputeAttention(currentQueryMap, currentKeyMap, currentAttentionScores, channelsPerHead_, attCPerHead_, inH_, inW_);
		//checkCUDNN(cudnnSoftmaxForward(cudnnHandle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, attentionDesc_, currentAttentionScores, &beta0, attentionDesc_, currentAttentionScores));
		SpatialSoftmaxHalf(currentAttentionScores, currentAttentionScores, batchSize_, attCPerHead_, inH_, inW_);
		cudaMemset(headOutput_, 0, outNCHW_*sizeof(__half));
		ApplyAttention(currentValueMap, currentAttentionScores, headOutput_, channelsPerHead_, attCPerHead_, inH_, inW_);
		checkCUDNN(cudnnAddTensor(cudnnHandle_, &headAlpha, outDesc_, headOutput_, &beta1, outDesc_, outData_));
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, outDesc_, inData_, &beta1, outDesc_, outData_));
	return outData_;
}
__half* SpatialAttentionLayer::Backward(__half* grad){
	cudaMemset(gradOut_, 0, outNCHW_*sizeof(__half));
	const float headAlpha = 1.0f/numHeads_;
	for(int h = 0; h<numHeads_; h++){
		const size_t weightOffset = h*dwWeightSizeSingle_;
		const size_t valueWeightOffset = h*pwWeightSizeSingle_;
		const size_t kqOffset = h*kqHeadSize_;
		const size_t vOffset = h*vHeadSize_;
		const __half* currentKeyMap = keyMap_ + kqOffset;
		const __half* currentQueryMap = queryMap_ + kqOffset;
		const __half* currentValueMap = valueMap_ + vOffset;
		const __half* currentAttentionScores = attentionScores_ + vOffset;
		__half* currentGradValueMap = gradValueMap_ + vOffset;
		__half* currentGradAttention = gradAttention_ + vOffset;
		__half* currentGradKeyMap = gradKeyMap_ + kqOffset;
		__half* currentGradQueryMap = gradQueryMap_ + kqOffset;
		ApplyAttentionBackward(grad, currentValueMap, currentAttentionScores, currentGradValueMap, currentGradAttention, channelsPerHead_, attCPerHead_, inH_, inW_);
		//checkCUDNN(cudnnSoftmaxBackward(cudnnHandle_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, attentionDesc_, currentAttentionScores, attentionDesc_, currentGradAttention, &beta0, attentionDesc_, currentGradAttention));
		SpatialSoftmaxBackwardHalf(currentAttentionScores, currentGradAttention, currentGradAttention, batchSize_, attCPerHead_, inH_, inW_);
		ComputeQueryKeyGrad(currentGradAttention, currentQueryMap, currentKeyMap, currentGradQueryMap, currentGradKeyMap, batchSize_, channelsPerHead_, attCPerHead_, inH_, inW_);
		checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &headAlpha, outDesc_, currentKeyMap, attentionDesc_, currentGradValueMap, valueConvDesc_, valueAlgos_.bwdFilterAlgo, valueWorkspace_, valueAlgos_.workspaceSize, &beta0, valueFilterDesc_, gradValueWeights_+valueWeightOffset));
		checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &headAlpha, valueFilterDesc_, valueWeights_+valueWeightOffset, attentionDesc_, currentGradValueMap, valueConvDesc_, valueAlgos_.bwdDataAlgo, valueWorkspace_, valueAlgos_.workspaceSize, &beta1, outDesc_, gradOut_));
		checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &headAlpha, outDesc_, inData_, outDesc_, currentGradQueryMap, keyQueryConvDesc_, keyQueryAlgos_.bwdFilterAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta0, keyQueryFilterDesc_, gradQueryWeights_+weightOffset));
		checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &headAlpha, keyQueryFilterDesc_, queryWeights_+weightOffset, outDesc_, currentGradQueryMap, keyQueryConvDesc_, keyQueryAlgos_.bwdDataAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta1, outDesc_, gradOut_));
		checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &headAlpha, outDesc_, inData_, outDesc_, currentGradKeyMap, keyQueryConvDesc_, keyQueryAlgos_.bwdFilterAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta0, keyQueryFilterDesc_, gradKeyWeights_+weightOffset));
		checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &headAlpha, keyQueryFilterDesc_, keyWeights_+weightOffset, outDesc_, currentGradKeyMap, keyQueryConvDesc_, keyQueryAlgos_.bwdDataAlgo, keyQueryWorkspace_, keyQueryAlgos_.workspaceSize, &beta1, outDesc_, gradOut_));
	}
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, outDesc_, grad, &beta1, outDesc_, gradOut_));
	return gradOut_;
}
void SpatialAttentionLayer::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWHalf(keyWeights_, gradKeyWeights_, m_Key_, v_Key_, learningRate, t_, weightDecay_, dwWeightSizeAll_);
		AdamWHalf(queryWeights_, gradQueryWeights_, m_Query_, v_Query_, learningRate, t_, weightDecay_, dwWeightSizeAll_);
		AdamWHalf(valueWeights_, gradValueWeights_, m_Pointwise_, v_Pointwise_, learningRate, t_, weightDecay_, pwWeightSizeAll_);
		++t_;
	} else{
		SGDHalf(keyWeights_, gradKeyWeights_, dwWeightSizeAll_, learningRate, weightDecay_);
		SGDHalf(queryWeights_, gradQueryWeights_, dwWeightSizeAll_, learningRate, weightDecay_);
		SGDHalf(valueWeights_, gradValueWeights_, pwWeightSizeAll_, learningRate, weightDecay_);
	}
}
void SpatialAttentionLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, keyWeights_, dwWeightSizeAll_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(buffer, queryWeights_, dwWeightSizeAll_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(buffer, valueWeights_, pwWeightSizeAll_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), pwWeightSizeAll_*sizeof(__half));
}
void SpatialAttentionLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(keyWeights_, buffer, dwWeightSizeAll_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(queryWeights_, buffer, dwWeightSizeAll_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), pwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(valueWeights_, buffer, pwWeightSizeAll_*sizeof(__half), cudaMemcpyHostToDevice);
}
void SpatialAttentionLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Key_, dwWeightSizeAll_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(buffer, v_Key_, dwWeightSizeAll_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(buffer, m_Query_, dwWeightSizeAll_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(buffer, v_Query_, dwWeightSizeAll_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(buffer, m_Pointwise_, pwWeightSizeAll_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), pwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(buffer, v_Pointwise_, pwWeightSizeAll_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), pwWeightSizeAll_*sizeof(__half));
}
void SpatialAttentionLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(m_Key_, buffer, dwWeightSizeAll_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(v_Key_, buffer, dwWeightSizeAll_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(m_Query_, buffer, dwWeightSizeAll_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), dwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(v_Query_, buffer, dwWeightSizeAll_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), pwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(m_Pointwise_, buffer, pwWeightSizeAll_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), pwWeightSizeAll_*sizeof(__half));
	cudaMemcpy(v_Pointwise_, buffer, pwWeightSizeAll_*sizeof(__half), cudaMemcpyHostToDevice);
}
size_t SpatialAttentionLayer::GetParameterSize(){
	return dwWeightSizeAll_*sizeof(__half);
}
size_t SpatialAttentionLayer::GetOptimizerStateSize(){
	return dwWeightSizeAll_*sizeof(__half);
}