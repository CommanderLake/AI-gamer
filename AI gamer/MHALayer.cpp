#include "MHALayer.h"
#include "common.h"
#include <iostream>
MultiHeadAttentionLayer::MultiHeadAttentionLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int seqLength, int embedSize, int numHeads, const char* layerName, bool train, float weightDecay) : cudnnHandle_(cudnnHandle),
	cublasHandle_(cublasHandle), batchSize_(batchSize), seqLength_(seqLength), embedSize_(embedSize), numHeads_(numHeads), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	weightSize_ = embedSize*embedSize;
	// Create descriptors
	checkCUDNN(cudnnCreateAttnDescriptor(&attnDesc_));
	checkCUDNN(cudnnCreateSeqDataDescriptor(&qDesc_));
	checkCUDNN(cudnnCreateSeqDataDescriptor(&kDesc_));
	checkCUDNN(cudnnCreateSeqDataDescriptor(&vDesc_));
	checkCUDNN(cudnnCreateSeqDataDescriptor(&oDesc_));
	// Set attention descriptor
	cudnnDropoutDescriptor_t attnDropoutDesc, postDropoutDesc;
	checkCUDNN(cudnnCreateDropoutDescriptor(&attnDropoutDesc));
	checkCUDNN(cudnnCreateDropoutDescriptor(&postDropoutDesc));
	checkCUDNN(cudnnSetAttnDescriptor(attnDesc_, CUDNN_ATTN_QUERYMAP_ALL_TO_ONE, // attnMode
		numHeads_, 1.0 / sqrt(embedSize / numHeads_), // smScaler
		CUDNN_DATA_HALF, CUDNN_DATA_FLOAT, // computePrec
		CUDNN_TENSOR_OP_MATH, attnDropoutDesc, postDropoutDesc, embedSize, // qSize
		embedSize, // kSize
		embedSize, // vSize
		embedSize, // qProjSize
		embedSize, // kProjSize
		embedSize, // vProjSize
		embedSize, // oProjSize
		seqLength, // qoMaxSeqLength
		seqLength, // kvMaxSeqLength
		batchSize, // maxBatchSize
		1)); // maxBeamSize
	// Set sequence data descriptors
	int dimA[4] = {batchSize, numHeads_, seqLength, embedSize/numHeads_};
	cudnnSeqDataAxis_t axes[4] = {CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_VECT_DIM};
	std::vector<int> seqLengthArray(batchSize_, seqLength_);
	checkCUDNN(cudnnSetSeqDataDescriptor(qDesc_, CUDNN_DATA_HALF, 4, dimA, axes, batchSize_, seqLengthArray.data(), nullptr));
	checkCUDNN(cudnnSetSeqDataDescriptor(kDesc_, CUDNN_DATA_HALF, 4, dimA, axes, batchSize_, seqLengthArray.data(), nullptr));
	checkCUDNN(cudnnSetSeqDataDescriptor(vDesc_, CUDNN_DATA_HALF, 4, dimA, axes, batchSize_, seqLengthArray.data(), nullptr));
	checkCUDNN(cudnnSetSeqDataDescriptor(oDesc_, CUDNN_DATA_HALF, 4, dimA, axes, batchSize_, seqLengthArray.data(), nullptr));
	// Allocate memory for weights and data
	CUDAMallocZero(&wq_, weightSize_*sizeof(__half));
	CUDAMallocZero(&wk_, weightSize_*sizeof(__half));
	CUDAMallocZero(&wv_, weightSize_*sizeof(__half));
	CUDAMallocZero(&wo_, weightSize_*sizeof(__half));
	// Initialize weights
	HeInit(wq_, weightSize_, embedSize);
	HeInit(wk_, weightSize_, embedSize);
	HeInit(wv_, weightSize_, embedSize);
	HeInit(wo_, weightSize_, embedSize);
	size_t dataSize = batchSize*seqLength*embedSize;
	CUDAMallocZero(&qData_, dataSize*sizeof(__half));
	CUDAMallocZero(&kData_, dataSize*sizeof(__half));
	CUDAMallocZero(&vData_, dataSize*sizeof(__half));
	CUDAMallocZero(&oData_, dataSize*sizeof(__half));
	if(train_){
		CUDAMallocZero(&gradWq_, weightSize_*sizeof(__half));
		CUDAMallocZero(&gradWk_, weightSize_*sizeof(__half));
		CUDAMallocZero(&gradWv_, weightSize_*sizeof(__half));
		CUDAMallocZero(&gradWo_, weightSize_*sizeof(__half));
		CUDAMallocZero(&gradIn_, dataSize*sizeof(__half));
		if(useAdamW_){
			CUDAMallocZero(&m_Wq_, weightSize_*sizeof(__half));
			CUDAMallocZero(&v_Wq_, weightSize_*sizeof(__half));
			CUDAMallocZero(&m_Wk_, weightSize_*sizeof(__half));
			CUDAMallocZero(&v_Wk_, weightSize_*sizeof(__half));
			CUDAMallocZero(&m_Wv_, weightSize_*sizeof(__half));
			CUDAMallocZero(&v_Wv_, weightSize_*sizeof(__half));
			CUDAMallocZero(&m_Wo_, weightSize_*sizeof(__half));
			CUDAMallocZero(&v_Wo_, weightSize_*sizeof(__half));
		}
	}
	// Get workspace size
	checkCUDNN(cudnnGetMultiHeadAttnBuffers(cudnnHandle_, attnDesc_, &workspaceSize_, &reserveSpaceSize_, nullptr));
	checkCUDA(cudaMalloc(&workspace_, workspaceSize_));
	checkCUDA(cudaMalloc(&reserveSpace_, reserveSpaceSize_));
	// Allocate device memory for sequence lengths
	checkCUDA(cudaMalloc(&d_SeqLengthsQo, batchSize_*sizeof(int)));
	checkCUDA(cudaMalloc(&d_SeqLengthsKv, batchSize_*sizeof(int)));
	std::vector<int> h_seqLengths(batchSize_, seqLength_);
	checkCUDA(cudaMemcpy(d_SeqLengthsQo, h_seqLengths.data(), batchSize_*sizeof(int), cudaMemcpyHostToDevice));
	checkCUDA(cudaMemcpy(d_SeqLengthsKv, h_seqLengths.data(), batchSize_*sizeof(int), cudaMemcpyHostToDevice));
	// Set up attention window
	loWinIdx = new std::vector<int>(seqLength_, 0);
	hiWinIdx = new std::vector<int>(seqLength_);
	for(int i = 0; i<seqLength_; ++i){
		(*hiWinIdx)[i] = i+1; // For self-attention
	}
	devSeqLengthsDqdo = new std::vector<int>(batchSize_, seqLength_);
	devSeqLengthsDkdv = new std::vector<int>(batchSize_, seqLength_);
}
MultiHeadAttentionLayer::~MultiHeadAttentionLayer(){
	cudaFree(wq_);
	cudaFree(wk_);
	cudaFree(wv_);
	cudaFree(wo_);
	cudaFree(qData_);
	cudaFree(kData_);
	cudaFree(vData_);
	cudaFree(oData_);
	cudaFree(workspace_);
	cudaFree(reserveSpace_);
	cudaFree(d_SeqLengthsQo);
	cudaFree(d_SeqLengthsKv);
	if(train_){
		cudaFree(gradWq_);
		cudaFree(gradWk_);
		cudaFree(gradWv_);
		cudaFree(gradWo_);
		cudaFree(gradIn_);
		if(useAdamW_){
			cudaFree(m_Wq_);
			cudaFree(v_Wq_);
			cudaFree(m_Wk_);
			cudaFree(v_Wk_);
			cudaFree(m_Wv_);
			cudaFree(v_Wv_);
			cudaFree(m_Wo_);
			cudaFree(v_Wo_);
		}
	}
	checkCUDNN(cudnnDestroyAttnDescriptor(attnDesc_));
	checkCUDNN(cudnnDestroySeqDataDescriptor(qDesc_));
	checkCUDNN(cudnnDestroySeqDataDescriptor(kDesc_));
	checkCUDNN(cudnnDestroySeqDataDescriptor(vDesc_));
	checkCUDNN(cudnnDestroySeqDataDescriptor(oDesc_));
}
__half* MultiHeadAttentionLayer::Forward(__half* data){
	const int currIdx = train_ ? -1 : 0;
	checkCUDNN(cudnnMultiHeadAttnForward(cudnnHandle_, attnDesc_,
		currIdx,
		(*loWinIdx).data(),
		(*hiWinIdx).data(),
		d_SeqLengthsQo,
		d_SeqLengthsKv,
		qDesc_, data,
		nullptr,  // residuals
		kDesc_, data,
		vDesc_, data,
		oDesc_, oData_,
		weightSize_*sizeof(__half), wq_,
		workspaceSize_, workspace_,
		train_ ? reserveSpaceSize_ : 0,
		train_ ? reserveSpace_ : nullptr));
	return oData_;
}
__half* MultiHeadAttentionLayer::Backward(__half* grad){
	// Backward data
	checkCUDNN(cudnnMultiHeadAttnBackwardData(cudnnHandle_, attnDesc_,
		(*loWinIdx).data(),
		(*hiWinIdx).data(),
		(*devSeqLengthsDqdo).data(),
		(*devSeqLengthsDkdv).data(),
		oDesc_, grad,
		qDesc_, gradIn_,
		qData_,
		kDesc_, gradWk_,
		kData_,
		vDesc_, gradWv_,
		vData_,
		weightSize_*sizeof(__half), wq_,
		workspaceSize_, workspace_,
		reserveSpaceSize_, reserveSpace_));
	// Backward weights
	checkCUDNN(cudnnMultiHeadAttnBackwardWeights(cudnnHandle_, attnDesc_,
		CUDNN_WGRAD_MODE_ADD,
		qDesc_, qData_,
		kDesc_, kData_,
		vDesc_, vData_,
		oDesc_, grad,
		weightSize_*sizeof(__half), wq_,
		gradWq_,
		workspaceSize_, workspace_,
		reserveSpaceSize_, reserveSpace_));
	return gradIn_;
}
void MultiHeadAttentionLayer::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWHalf(wq_, gradWq_, m_Wq_, v_Wq_, learningRate, t_, weightDecay_, weightSize_);
		AdamWHalf(wk_, gradWk_, m_Wk_, v_Wk_, learningRate, t_, weightDecay_, weightSize_);
		AdamWHalf(wv_, gradWv_, m_Wv_, v_Wv_, learningRate, t_, weightDecay_, weightSize_);
		AdamWHalf(wo_, gradWo_, m_Wo_, v_Wo_, learningRate, t_, weightDecay_, weightSize_);
		++t_;
	} else{
		SGDHalf(wq_, gradWq_, weightSize_, learningRate);
		SGDHalf(wk_, gradWk_, weightSize_, learningRate);
		SGDHalf(wv_, gradWv_, weightSize_, learningRate);
		SGDHalf(wo_, gradWo_, weightSize_, learningRate);
	}
}
void MultiHeadAttentionLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, wq_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
	cudaMemcpy(buffer, wk_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
	cudaMemcpy(buffer, wv_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
	cudaMemcpy(buffer, wo_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
}
void MultiHeadAttentionLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
	cudaMemcpy(wq_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
	cudaMemcpy(wk_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
	cudaMemcpy(wv_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
	cudaMemcpy(wo_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
}
void MultiHeadAttentionLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(useAdamW_){
		cudaMemcpy(buffer, m_Wq_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(buffer, v_Wq_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(buffer, m_Wk_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(buffer, v_Wk_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(buffer, m_Wv_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(buffer, v_Wv_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(buffer, m_Wo_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(buffer, v_Wo_, weightSize_*sizeof(__half), cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_*sizeof(__half));
	}
}
void MultiHeadAttentionLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(useAdamW_){
		file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(m_Wq_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(v_Wq_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(m_Wk_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(v_Wk_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(m_Wv_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(v_Wv_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(m_Wo_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), weightSize_*sizeof(__half));
		cudaMemcpy(v_Wo_, buffer, weightSize_*sizeof(__half), cudaMemcpyHostToDevice);
	}
}
size_t MultiHeadAttentionLayer::GetParameterSize(){
	return 4*weightSize_*sizeof(__half); // wq, wk, wv, wo
}
size_t MultiHeadAttentionLayer::GetOptimizerStateSize(){
	return useAdamW_ ? 8*weightSize_*sizeof(__half) : 0; // m and v for wq, wk, wv, wo
}