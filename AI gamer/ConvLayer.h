#pragma once
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
class ConvLayer final : public Layer{
public:
	ConvLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int bitchSize, int inputChannels, int outputChannels, int filterSize, int stride, int padding, int* width, int* height, int* inputCHW, int outW, int outH, const char* layerName);
	~ConvLayer() override;
	__half* forward(__half* data) override;
	__half* backward(__half* grad) override;
	void updateParameters(float learningRate) override;
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	cudnnTensorDescriptor_t inDesc_;
	cudnnFilterDescriptor_t filterDesc_;
	cudnnConvolutionDescriptor_t convDesc_;
	cudnnConvolutionFwdAlgo_t fwdAlgo_;
	cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo_;
	cudnnConvolutionBwdDataAlgo_t bwdDataAlgo_;
	cudnnTensorDescriptor_t biasDesc_;
	cudnnTensorDescriptor_t inGradDesc_;
	cudnnTensorDescriptor_t outGradDesc_;
	size_t bitchSize_;
	size_t inC_;
	const size_t inCHW_;
	size_t outC_;
	size_t outCHW_;
	size_t weightCount_;
	size_t gradOutSize_;
	const __half* inData_;
	__half* outData_;
	__half* bias_;
	__half* weights_;
	__half* gradOut_;
	__half* gradWeights_;
	__half* gradBias_;
	size_t workspaceSize_ = 0;
	void* workspace_;
	float *m_weights_, *v_weights_;
	float *m_bias_, *v_bias_;
	int t_ = 1;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
};