#pragma once
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
class FCLayer final : public Layer{
public:
	FCLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int bitchSize, int inputSize, int outputSize, const char* layername);
	~FCLayer() override;
	__half* forward(__half* data) override;
	__half* backward(__half* grad) override;
	void updateParameters(float learningRate) override;
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	cudnnTensorDescriptor_t inDesc_;
	size_t bitchSize_;
	size_t inC_;
	size_t outC_;
	size_t weightCount_;
	size_t gradOutSize_;
	const __half* inData_;
	__half* outData_;
	__half* weights_;
	__half* bias_;
	__half* gradOut_;
	__half* gradWeights_;
	__half* gradBias_;
	float epsilon_;
	float *m_weights_, *v_weights_;
	float *m_bias_, *v_bias_;
	int t_ = 1;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
};