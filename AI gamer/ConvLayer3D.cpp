#include "ConvLayer3D.h"
#include "common.h"
ConvLayer3D::ConvLayer3D(cudnnHandle_t cudnnHandle, int batchSize, int inChannels, int outChannels, int filterD, int filterHW, int strideD, int strideHW, int padD, int padHW, int* width, int* height, int* depth, const char* layerName, bool train,
						float weightDecay) : cudnnHandle_(cudnnHandle), fwdAlgo_(), bwdFilterAlgo_(), bwdDataAlgo_(), batchSize_(batchSize), inC_(inChannels), outC_(outChannels), inData_(nullptr),
											outData_(nullptr), weights_(nullptr), weightDecay_(weightDecay){
	layerName_ = layerName;
	train_ = train;
	int inWidth = *width, inHeight = *height, inDepth = *depth, outWidth, outHeight, outDepth;
	// Create descriptors
	checkCUDNN(cudnnCreateTensorDescriptor(&inDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&outDesc_));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc_));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&biasDesc_));
	// Set up bias descriptor (remains 4D)
	int biasDims[5] = {1, outC_, 1, 1, 1};
	int biasStrides[5] = {outC_, 1, 1, 1, 1};
	checkCUDNN(cudnnSetTensorNdDescriptor(biasDesc_, CUDNN_DATA_HALF, 5, biasDims, biasStrides));
	// Set up input tensor descriptor (5D)
	int inDims[5] = {batchSize_, inC_, inDepth, inHeight, inWidth};
	int inStrides[5] = {inC_*inDepth*inHeight*inWidth, inDepth*inHeight*inWidth, inHeight*inWidth, inWidth, 1};
	checkCUDNN(cudnnSetTensorNdDescriptor(inDesc_, CUDNN_DATA_HALF, 5, inDims, inStrides));
	// Set up filter descriptor (5D)
	int filterDims[5] = {outC_, inC_, filterD, filterHW, filterHW};
	checkCUDNN(cudnnSetFilterNdDescriptor(filterDesc_, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, 5, filterDims));
	// Set up convolution descriptor
	int padA[3] = {padD, padHW, padHW};
	int filterStrideA[3] = {strideD, strideHW, strideHW};
	int dilationA[3] = {1, 1, 1};
	checkCUDNN(cudnnSetConvolutionNdDescriptor(convDesc_, 3, padA, filterStrideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));
	checkCUDNN(cudnnSetConvolutionMathType(convDesc_, CUDNN_TENSOR_OP_MATH));//S
	// Get output dimensions
	int outDims[5];
	checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc_, inDesc_, filterDesc_, 5, outDims));
	outDepth = outDims[2];
	outHeight = outDims[3];
	outWidth = outDims[4];
	// Set up output tensor descriptor (5D)
	int outStrides[5] = {outC_*outDepth*outHeight*outWidth, outDepth*outHeight*outWidth, outHeight*outWidth, outWidth, 1};
	checkCUDNN(cudnnSetTensorNdDescriptor(outDesc_, CUDNN_DATA_HALF, 5, outDims, outStrides));
	outNCHW_ = outWidth*outHeight*outDepth*outC_*batchSize_;
	gradOutSize_ = inWidth*inHeight*inDepth*inC_*batchSize_*sizeof(__half);
	const auto fanIn = inC_*filterD*filterHW*filterHW;
	weightCount_ = outC_*fanIn;
	CUDAMallocZero(&outData_, outNCHW_*sizeof(__half));
	CUDAMallocZero(&weights_, weightCount_*sizeof(__half));
	CUDAMallocZero(&bias_, outC_*sizeof(__half));
	if(train_){
		HeInit(weights_, weightCount_, fanIn);
		checkCUDNN(cudnnCreateTensorDescriptor(&outGradDesc_));
		checkCUDNN(cudnnCreateTensorDescriptor(&inGradDesc_));
		checkCUDNN(cudnnSetTensorNdDescriptor(outGradDesc_, CUDNN_DATA_HALF, 5, inDims, inStrides));
		checkCUDNN(cudnnSetTensorNdDescriptor(inGradDesc_, CUDNN_DATA_HALF, 5, outDims, outStrides));
		CUDAMallocZero(&gradWeights_, weightCount_*sizeof(__half));
		CUDAMallocZero(&gradBias_, outC_*sizeof(__half));
		CUDAMallocZero(&gradOut_, gradOutSize_);
		if(useAdamW_){
			CUDAMallocZero(&m_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&v_Weights_, weightCount_*sizeof(__half));
			CUDAMallocZero(&m_Bias_, outC_*sizeof(__half));
			CUDAMallocZero(&v_Bias_, outC_*sizeof(__half));
		}
	}
	cudnnConvolutionFwdAlgoPerf_t fwdAlgoPerf[10];
	cudnnConvolutionBwdFilterAlgoPerf_t bwdFilterAlgoPerf[10];
	cudnnConvolutionBwdDataAlgoPerf_t bwdDataAlgoPerf[10];
	int returnedAlgoCount;
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_, inDesc_, filterDesc_, convDesc_, outDesc_, 10, &returnedAlgoCount, fwdAlgoPerf));
	fwdAlgo_ = fwdAlgoPerf[0].algo;
	const size_t fwdWorkspaceSize = fwdAlgoPerf[0].memory;
	checkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle_, inDesc_, outDesc_, convDesc_, filterDesc_, 10, &returnedAlgoCount, bwdFilterAlgoPerf));
	bwdFilterAlgo_ = bwdFilterAlgoPerf[0].algo;
	const size_t bwdFilterWorkspaceSize = bwdFilterAlgoPerf[0].memory;
	checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle_, filterDesc_, outDesc_, convDesc_, inDesc_, 10, &returnedAlgoCount, bwdDataAlgoPerf));
	bwdDataAlgo_ = bwdDataAlgoPerf[0].algo;
	const size_t bwdDataWorkspaceSize = bwdDataAlgoPerf[0].memory;
	workspaceSize_ = std::max(fwdWorkspaceSize, std::max(bwdFilterWorkspaceSize, bwdDataWorkspaceSize));
	checkCUDA(cudaMalloc(&workspace_, workspaceSize_));
	*width = outWidth;
	*height = outHeight;
	*depth = outDepth;
}
ConvLayer3D::~ConvLayer3D(){
	cudaFree(outData_);
	cudaFree(weights_);
	cudaFree(bias_);
	cudaFree(workspace_);
	checkCUDNN(cudnnDestroyTensorDescriptor(inDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(outDesc_));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc_));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc_));
	checkCUDNN(cudnnDestroyTensorDescriptor(biasDesc_));
	if(train_){
		cudaFree(gradWeights_);
		cudaFree(gradBias_);
		cudaFree(gradOut_);
		if(useAdamW_){
			cudaFree(m_Weights_);
			cudaFree(v_Weights_);
			cudaFree(m_Bias_);
			cudaFree(v_Bias_);
		}
		checkCUDNN(cudnnDestroyTensorDescriptor(outGradDesc_));
	}
}
__half* ConvLayer3D::Forward(__half* data){
	inData_ = data;
	checkCUDNN(cudnnConvolutionForward(cudnnHandle_, &alpha, inDesc_, data, filterDesc_, weights_, convDesc_, fwdAlgo_, workspace_, workspaceSize_, &beta0, outDesc_, outData_));
	checkCUDNN(cudnnAddTensor(cudnnHandle_, &alpha, biasDesc_, bias_, &beta1, outDesc_, outData_));
	return outData_;
}
__half* ConvLayer3D::Backward(__half* grad){
	checkCUDNN(cudnnConvolutionBackwardFilter(cudnnHandle_, &alpha, inDesc_, inData_, inGradDesc_, grad, convDesc_, bwdFilterAlgo_, workspace_, workspaceSize_, &beta0, filterDesc_, gradWeights_));
	checkCUDNN(cudnnConvolutionBackwardBias(cudnnHandle_, &alpha, inGradDesc_, grad, &beta0, biasDesc_, gradBias_));
	checkCUDNN(cudnnConvolutionBackwardData(cudnnHandle_, &alpha, filterDesc_, weights_, inGradDesc_, grad, convDesc_, bwdDataAlgo_, workspace_, workspaceSize_, &beta0, outGradDesc_, gradOut_));
	return gradOut_;
}
void ConvLayer3D::UpdateParameters(float learningRate){
	if(useAdamW_){
		AdamWHalf(weights_, gradWeights_, m_Weights_, v_Weights_, learningRate, t_, weightDecay_, weightCount_);
		AdamWHalf(bias_, gradBias_, m_Bias_, v_Bias_, learningRate, t_, weightDecay_, outC_);
		++t_;
	} else{
		SGDHalf(weights_, gradWeights_, weightCount_, learningRate);
		SGDHalf(bias_, gradBias_, outC_, learningRate);
	}
}
void ConvLayer3D::SaveParameters(std::ofstream& file, unsigned char* buffer){
	cudaMemcpy(buffer, weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
}
void ConvLayer3D::LoadParameters(std::ifstream& file, unsigned char* buffer){
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
}
void ConvLayer3D::SaveOptimizerState(std::ofstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	cudaMemcpy(buffer, m_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, v_Weights_, weightCount_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(buffer, m_Bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(buffer, v_Bias_, outC_*sizeof(__half), cudaMemcpyDeviceToHost);
	file.write(reinterpret_cast<const char*>(buffer), outC_*sizeof(__half));
}
void ConvLayer3D::LoadOptimizerState(std::ifstream& file, unsigned char* buffer){
	if(!useAdamW_) return;
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(m_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), weightCount_*sizeof(__half));
	cudaMemcpy(v_Weights_, buffer, weightCount_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(m_Bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
	file.read(reinterpret_cast<char*>(buffer), outC_*sizeof(__half));
	cudaMemcpy(v_Bias_, buffer, outC_*sizeof(__half), cudaMemcpyHostToDevice);
}
size_t ConvLayer3D::GetParameterSize(){ return weightCount_*sizeof(__half); }
size_t ConvLayer3D::GetOptimizerStateSize(){ return weightCount_*sizeof(__half); }