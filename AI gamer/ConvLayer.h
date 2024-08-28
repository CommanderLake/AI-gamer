#pragma once
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
class ConvLayer final : public Layer{
public:
	const bool useAdamW_ = true;
	ConvLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int bitchSize, int inputChannels, int outputChannels, int filterSize, int stride, int padding, int* width, int* height, int* inputCHW, const char* layerName, bool train, float weightDecay);
	~ConvLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float learningRate) override;
	void SaveParameters(std::ofstream& file, unsigned char* buffer) override;
	void LoadParameters(std::ifstream& file, unsigned char* buffer) override;
	void SaveOptimizerState(std::ofstream& file, unsigned char* buffer) override;
	void LoadOptimizerState(std::ifstream& file, unsigned char* buffer) override;
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
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
	size_t gradOutSize_;
	__half* inData_;
	__half* outData_;
	__half* bias_;
	__half* weights_;
	__half* gradOut_;
	__half* gradWeights_;
	__half* gradBias_;
	size_t workspaceSize_ = 0;
	void* workspace_;
	__half *m_Weights_, *v_Weights_;
	__half *m_Bias_, *v_Bias_;
	int t_ = 1;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
	float weightDecay_;
};