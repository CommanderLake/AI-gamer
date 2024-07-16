#pragma once
#include "Layer.h"
#include <cudnn.h>
class BatchNorm final : public Layer{
public:
	BatchNorm(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, int bitchSize);
	~BatchNorm() override;
	__half* forward(__half* data) override;
	__half* backward(__half* grad) override;
	void updateParameters(float learningRate) override;
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t bnScaleBiasDesc_;
	size_t bitchSize_;
	__half* inData_;
	__half* outData_;
	float epsilon_ = 1e-6;
	float* bnScale_ = nullptr;
	float* bnBias_ = nullptr;
	float* gradBnScale_ = nullptr;
	float* gradBnBias_ = nullptr;
	float* bnRunningMean_ = nullptr;
	float* bnRunningVar_ = nullptr;
	float* bnSavedMean_ = nullptr;
	float* bnSavedInvVariance_ = nullptr;
	float *m_bnScale_, *v_bnScale_;
	float *m_bnBias_, *v_bnBias_;
	int t_ = 1;
	int outC_;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
};