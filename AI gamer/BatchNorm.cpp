#include "BatchNorm.h"
#include "common.h"
BatchNorm::BatchNorm(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, int bitchSize): cudnnHandle_(cudnnHandle), bitchSize_(bitchSize), inData_(nullptr){
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc_, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outSize_ = n*c*h*w;
	outC_ = c;
	checkCUDNN(cudnnCreateTensorDescriptor(&bnScaleBiasDesc_));
	checkCUDNN(cudnnDeriveBNTensorDescriptor(bnScaleBiasDesc_, outDesc_, CUDNN_BATCHNORM_SPATIAL));
	checkCUDA(cudaMalloc(&outData_, outSize_*sizeof(__half)));
	checkCUDA(cudaMemset(outData_, 0, outSize_*sizeof(__half)));
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
	checkCUDA(cudaMalloc(&m_bnScale_, outC_*sizeof(float)));
	checkCUDA(cudaMalloc(&v_bnScale_, outC_*sizeof(float)));
	checkCUDA(cudaMalloc(&m_bnBias_, outC_*sizeof(float)));
	checkCUDA(cudaMalloc(&v_bnBias_, outC_*sizeof(float)));
	checkCUDA(cudaMemset(m_bnScale_, 0, outC_*sizeof(float)));
	checkCUDA(cudaMemset(v_bnScale_, 0, outC_*sizeof(float)));
	checkCUDA(cudaMemset(m_bnBias_, 0, outC_*sizeof(float)));
	checkCUDA(cudaMemset(v_bnBias_, 0, outC_*sizeof(float)));
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
__half* BatchNorm::forward(__half* data){
	inData_ = data;
	checkCUDNN(
		cudnnBatchNormalizationForwardTraining(cudnnHandle_, CUDNN_BATCHNORM_SPATIAL, &alpha, &beta0, outDesc_, data, outDesc_, outData_, bnScaleBiasDesc_, bnScale_, bnBias_, 1.0, bnRunningMean_, bnRunningVar_, epsilon_,
			bnSavedMean_, bnSavedInvVariance_));
	return outData_;
}
__half* BatchNorm::backward(__half* grad){
	checkCUDNN(cudnnBatchNormalizationBackward(
		cudnnHandle_,
		CUDNN_BATCHNORM_SPATIAL,
		&alpha, &beta0, &alpha, &beta1,
		outDesc_, inData_,	// x
		outDesc_, grad,		// dy (backpropagated gradient input)
		outDesc_, grad,		// dx (gradient output)
		bnScaleBiasDesc_,	// dBnScaleBiasDesc
		bnScale_,			// bnScaleData (batch normalization scale parameter)
		gradBnScale_,		// dBnScaleData (gradients of bnScaleData)
		gradBnBias_,		// dBnBiasData (gradients of bnBiasData)
		epsilon_,
		bnSavedMean_, bnSavedInvVariance_));
	return grad;
}
void BatchNorm::updateParameters(float learningRate){
	AdamWFloat(bnScale_, m_bnScale_, v_bnScale_, learningRate, gradBnScale_, outC_, t_, 0.01F);
	AdamWFloat(bnBias_, m_bnBias_, v_bnBias_, learningRate, gradBnBias_, outC_, t_, 0.01F);
	++t_;
}