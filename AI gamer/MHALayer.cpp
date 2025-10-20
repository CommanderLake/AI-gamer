#include "MHALayer.h"
#include "common.h"
#include "CuCommon.cuh"
#include <iostream>
MultiHeadAttentionLayer::MultiHeadAttentionLayer(const cudnnHandle_t cudnnHandle, const int batchSize, const int timeSize, const int vectorSize, const int numHeads, const char* layerName, const bool train, const float weightDecay, const int gradAccumLength) : cudnnHandle_(cudnnHandle), batchSize_(batchSize), timeSize_(timeSize), vectorSize_(vectorSize), numHeads_(numHeads), inData_(nullptr), weightDecay_(weightDecay), gradAccumLength_(gradAccumLength){
	layerName_ = layerName;
	train_ = train;
	checkCUDNN(cudnnCreateAttnDescriptor(&attnDesc_));
	checkCUDNN(cudnnCreateSeqDataDescriptor(&qkvDesc_));
	checkCUDNN(cudnnCreateSeqDataDescriptor(&outDesc_));
	const int dk = vectorSize_ / numHeads_;
	const double smScaler = 1.0 / std::sqrt(static_cast<double>(dk));
	checkCUDNN(cudnnSetAttnDescriptor(attnDesc_, CUDNN_ATTN_QUERYMAP_ONE_TO_ONE, numHeads_, smScaler, CUDNN_DATA_HALF, CUDNN_DATA_HALF, CUDNN_TENSOR_OP_MATH, nullptr, nullptr,
		vectorSize_, // qSize
		vectorSize_, // kSize
		vectorSize_, // vSize
		dk, // qProjSize
		dk, // kProjSize
		dk, // vProjSize
		vectorSize_, // oProjSize
		timeSize_, // qoMaxSeqLength
		timeSize_, // kvMaxSeqLength
		batchSize_, // maxBatchSize
		1)); // maxBeamSize
	SetFineTune(train);
	checkCUDNN(cudnnGetMultiHeadAttnBuffers(cudnnHandle_, attnDesc_, &weightSize_, &workspaceSize_, &reserveSpaceSize_));
	CUDAMallocZero(&weights_, weightSize_);
	cudnnTensorDescriptor_t wDesc;
	checkCUDNN(cudnnCreateTensorDescriptor(&wDesc));
	auto initWeight = [&](const cudnnMultiHeadAttnWeightKind_t kind){
		void* addr = nullptr;
		checkCUDNN(cudnnGetMultiHeadAttnWeights(cudnnHandle_, attnDesc_, kind, weightSize_, weights_, wDesc, &addr));
		if(!addr) return;
		cudnnDataType_t dt;
		auto ndims = 3;
		int d[3], s[3];
		checkCUDNN(cudnnGetTensorNdDescriptor(wDesc, ndims, &dt, &ndims, d, s));
		size_t elems = 1;
		for(int i = 0; i < ndims; ++i){
			elems *= static_cast<size_t>(d[i]);
		}
		const int fanIn = d[ndims - 1];
		const int fanOut = ndims > 1 ? static_cast<int>(elems / static_cast<size_t>(fanIn)) : fanIn;
		WeightInit(static_cast<__half*>(addr), static_cast<int>(elems), fanIn, fanOut, Xavier);
	};
	initWeight(CUDNN_MH_ATTN_Q_WEIGHTS);
	initWeight(CUDNN_MH_ATTN_K_WEIGHTS);
	initWeight(CUDNN_MH_ATTN_V_WEIGHTS);
	initWeight(CUDNN_MH_ATTN_O_WEIGHTS);
	cudnnDestroyTensorDescriptor(wDesc);
	outNCHW_ = batchSize_*timeSize_*vectorSize_;
	alphaWeights_ = gradAccumLength_ > 0 ? 1.0f/(static_cast<float>(batchSize_)*timeSize_*gradAccumLength_) : 1.0f;
	const size_t dataSize = outNCHW_;
	CUDAMallocZero(&outData_, dataSize*sizeof(__half));
	if(train_){
		CUDAMallocZero(&gradWeights_, weightSize_);
		CUDAMallocZero(&gradKeys_, dataSize*sizeof(__half));
		CUDAMallocZero(&gradValues_, dataSize*sizeof(__half));
		CUDAMallocZero(&outGrad_, dataSize*sizeof(__half));
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightSize_);
			CUDAMallocZero(&v_Weights_, weightSize_);
		}
	}
	checkCUDA(cudaMalloc(&workspace_, workspaceSize_));
	checkCUDA(cudaMalloc(&reserveSpace_, reserveSpaceSize_));
	checkCUDA(cudaMalloc(&d_SeqLengths, batchSize_*sizeof(int)));
	const std::vector<int> h_seqLengths(batchSize_, timeSize_);
	checkCUDA(cudaMemcpy(d_SeqLengths, h_seqLengths.data(), batchSize_*sizeof(int), cudaMemcpyHostToDevice));
	loWinIdx = new std::vector<int>(timeSize_, 0);
	hiWinIdx = new std::vector<int>(timeSize_, timeSize_);
}
MultiHeadAttentionLayer::~MultiHeadAttentionLayer(){
	delete loWinIdx;
	delete hiWinIdx;
	cudaFree(weights_);
	cudaFree(outData_);
	cudaFree(workspace_);
	cudaFree(reserveSpace_);
	cudaFree(d_SeqLengths);
	if(train_){
		cudaFree(gradWeights_);
		cudaFree(gradKeys_);
		cudaFree(gradValues_);
		cudaFree(outGrad_);
		if(useAdamW_){
			cudaFree(m_Weights_);
			cudaFree(v_Weights_);
		}
	}
	checkCUDNN(cudnnDestroyAttnDescriptor(attnDesc_));
	checkCUDNN(cudnnDestroySeqDataDescriptor(qkvDesc_));
	checkCUDNN(cudnnDestroySeqDataDescriptor(outDesc_));
}
__half* MultiHeadAttentionLayer::Forward(__half* data){
	inData_ = data;
	const int currIdx = train_ ? -1 : 0;
	checkCUDNN(cudnnMultiHeadAttnForward(cudnnHandle_, attnDesc_, currIdx, (*loWinIdx).data(), (*hiWinIdx).data(), d_SeqLengths, d_SeqLengths, qkvDesc_, data, nullptr, qkvDesc_, data, qkvDesc_, data, outDesc_, outData_, weightSize_, weights_, workspaceSize_, workspace_, train_ ? reserveSpaceSize_ : 0, train_ ? reserveSpace_ : nullptr));
	return outData_;
}
__half* MultiHeadAttentionLayer::Backward(__half* grad){
	if(!train_){
		return grad;
	}
	const int prevCount = accumCount_++;
	checkCUDNN(cudnnMultiHeadAttnBackwardData(cudnnHandle_, attnDesc_, (*loWinIdx).data(), (*hiWinIdx).data(), d_SeqLengths, d_SeqLengths, outDesc_, grad, qkvDesc_, outGrad_, inData_, qkvDesc_, gradKeys_, inData_, qkvDesc_, gradValues_, inData_, weightSize_, weights_, workspaceSize_, workspace_, reserveSpaceSize_, reserveSpace_));
	const int accumLen = gradAccumLength_ > 0 ? gradAccumLength_ : 1;
	const auto wgradMode = prevCount % accumLen == 0 ? CUDNN_WGRAD_MODE_SET : CUDNN_WGRAD_MODE_ADD;
	checkCUDNN(cudnnMultiHeadAttnBackwardWeights(cudnnHandle_, attnDesc_, wgradMode, qkvDesc_, inData_, qkvDesc_, inData_, qkvDesc_, inData_, outDesc_, grad, weightSize_, weights_, gradWeights_, workspaceSize_, workspace_, reserveSpaceSize_, reserveSpace_));
	return outGrad_;
}
void MultiHeadAttentionLayer::UpdateParameters(float learningRate){
	if(!train_) return;
	if(gradAccumLength_ <= 0) return;
	if(accumCount_%gradAccumLength_>0) return;
	ScaleArrayHalf(gradWeights_, weightSize_/sizeof(__half), alphaWeights_);
	if(useAdamW_){
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, weightSize_/sizeof(__half));
		++t_;
	} else{ SGDHalf(weights_, gradWeights_, weightSize_/sizeof(__half), learningRate, weightDecay_); }
}
void MultiHeadAttentionLayer::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightSize_, cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightSize_);
}
void MultiHeadAttentionLayer::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightSize_);
	cudaMemcpy(weights_, buffer, weightSize_, cudaMemcpyHostToDevice);
}
void MultiHeadAttentionLayer::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(useAdamW_){
		cudaMemcpy(buffer, m_Weights_, weightSize_, cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_);
		cudaMemcpy(buffer, v_Weights_, weightSize_, cudaMemcpyDeviceToHost);
		file.write(reinterpret_cast<const char*>(buffer), weightSize_);
	}
}
void MultiHeadAttentionLayer::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(useAdamW_){
		file.read(reinterpret_cast<char*>(buffer), weightSize_);
		cudaMemcpy(m_Weights_, buffer, weightSize_, cudaMemcpyHostToDevice);
		file.read(reinterpret_cast<char*>(buffer), weightSize_);
		cudaMemcpy(v_Weights_, buffer, weightSize_, cudaMemcpyHostToDevice);
	}
}
size_t MultiHeadAttentionLayer::GetParameterSize(){ return weightSize_; }
size_t MultiHeadAttentionLayer::GetOptimizerStateSize(){
	return useAdamW_ ? 2*weightSize_ : 0;
}
void MultiHeadAttentionLayer::SetFineTune(const bool enable){
	int bs;
	if(enable){
		train_ = true;
		bs = batchSize_;
	} else{
		train_ = false;
		bs = 1;
	}
	outNCHW_ = bs*timeSize_*vectorSize_;
	alphaWeights_ = gradAccumLength_ > 0 ? 1.0f/(static_cast<float>(bs)*timeSize_*gradAccumLength_) : 1.0f;
	int dimA[CUDNN_SEQDATA_DIM_COUNT];
	dimA[CUDNN_SEQDATA_TIME_DIM] = timeSize_;
	dimA[CUDNN_SEQDATA_BATCH_DIM] = bs;
	dimA[CUDNN_SEQDATA_BEAM_DIM] = 1;
	dimA[CUDNN_SEQDATA_VECT_DIM] = vectorSize_;
	const cudnnSeqDataAxis_t axes[4] = {CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM};
	const std::vector<int> seqLengthArray(bs, timeSize_);
	checkCUDNN(cudnnSetSeqDataDescriptor(qkvDesc_, CUDNN_DATA_HALF, 4, dimA, axes, seqLengthArray.size(), seqLengthArray.data(), nullptr));
	checkCUDNN(cudnnSetSeqDataDescriptor(outDesc_, CUDNN_DATA_HALF, 4, dimA, axes, seqLengthArray.size(), seqLengthArray.data(), nullptr));
}