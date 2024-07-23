#include "BatchNorm.h"
#include "common.h"
#include <vector>
BatchNorm::BatchNorm(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, cudnnBatchNormMode_t bnMode, int bitchSize, const char* layerName): cudnnHandle_(cudnnHandle), bnMode_(bnMode), bitchSize_(bitchSize), inData_(nullptr), epsilon_(1e-6){
	layerName_ = layerName;
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outNCHW_ = n*c*h*w;
	outC_ = c;
	checkCUDNN(cudnnCreateTensorDescriptor(&gradDesc_));
	checkCUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasDesc_));
	checkCUDNN(cudnnSetTensor4dDescriptor(gradDesc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h, w));
	checkCUDNN(cudnnDeriveBNTensorDescriptor(bnScaleBiasDesc_, outDesc, bnMode_));
	checkCUDA(cudaMalloc(&outData_, outNCHW_*sizeof(__half)));
	checkCUDA(cudaMemset(outData_, 0, outNCHW_*sizeof(__half)));
	const auto bnSizeBytes = outC_ * sizeof(float);
	checkCUDA(cudaMalloc(&bnScale_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnBias_, bnSizeBytes));
	checkCUDA(cudaMalloc(&gradBnScale_, bnSizeBytes));
	checkCUDA(cudaMalloc(&gradBnBias_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnRunningMean_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnRunningVar_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnSavedMean_, bnSizeBytes));
	checkCUDA(cudaMalloc(&bnSavedInvVariance_, bnSizeBytes));
	const std::vector<float> bnScaleInit(outC_, 1.0f);
	checkCUDA(cudaMemcpy(bnScale_, bnScaleInit.data(), bnSizeBytes, cudaMemcpyHostToDevice));
	checkCUDA(cudaMemset(bnBias_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(gradBnScale_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(gradBnBias_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(bnRunningMean_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(bnRunningVar_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(bnSavedMean_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(bnSavedInvVariance_, 0, bnSizeBytes));
	checkCUDA(cudaMalloc(&m_bnScale_, bnSizeBytes));
	checkCUDA(cudaMalloc(&v_bnScale_, bnSizeBytes));
	checkCUDA(cudaMalloc(&m_bnBias_, bnSizeBytes));
	checkCUDA(cudaMalloc(&v_bnBias_, bnSizeBytes));
	checkCUDA(cudaMemset(m_bnScale_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(v_bnScale_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(m_bnBias_, 0, bnSizeBytes));
	checkCUDA(cudaMemset(v_bnBias_, 0, bnSizeBytes));
}
BatchNorm::~BatchNorm(){
	cudaFree(bnScale_);
	cudaFree(bnBias_);
	cudaFree(gradBnScale_);
	cudaFree(gradBnBias_);
	cudaFree(bnRunningMean_);
	cudaFree(bnRunningVar_);
	cudaFree(bnSavedMean_);
	cudaFree(bnSavedInvVariance_);
	cudaFree(m_bnScale_);
	cudaFree(v_bnScale_);
	cudaFree(m_bnBias_);
	cudaFree(v_bnBias_);
	checkCUDNN(cudnnDestroyTensorDescriptor(bnScaleBiasDesc_));
}
__half* BatchNorm::Forward(__half* data){
	inData_ = data;
	checkCUDNN(cudnnBatchNormalizationForwardTraining(cudnnHandle_, bnMode_, &alpha, &beta0, outDesc_, data, outDesc_, outData_, bnScaleBiasDesc_, bnScale_, bnBias_, 0.1, bnRunningMean_, bnRunningVar_, epsilon_, bnSavedMean_, bnSavedInvVariance_));
	return outData_;
}
__half* BatchNorm::Backward(__half* grad){
	checkCUDNN(cudnnBatchNormalizationBackward(
		cudnnHandle_,
		bnMode_,
		&alpha, &beta0, &alpha, &beta1,
		outDesc_, inData_,	// x
		gradDesc_, grad,	// dy (backpropagated gradient input)
		gradDesc_, grad,	// dx (gradient output)
		bnScaleBiasDesc_,	// dBnScaleBiasDesc
		bnScale_,			// bnScaleData (batch normalization scale parameter)
		gradBnScale_,		// dBnScaleData (gradients of bnScaleData)
		gradBnBias_,		// dBnBiasData (gradients of bnBiasData)
		epsilon_,
		bnSavedMean_, bnSavedInvVariance_));
	return grad;
}
void BatchNorm::UpdateParameters(float learningRate){
	//SGDFloat(bnScale_, learningRate, gradBnScale_, outC_);
	//SGDFloat(bnBias_, learningRate, gradBnBias_, outC_);
	AdamWFloat(bnScale_, m_bnScale_, v_bnScale_, learningRate, gradBnScale_, outC_, t_, 0.0001F);
	AdamWFloat(bnBias_, m_bnBias_, v_bnBias_, learningRate, gradBnBias_, outC_, t_, 0.0001F);
	++t_;
}