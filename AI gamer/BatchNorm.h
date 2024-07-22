#pragma once
#include "Layer.h"
#include <cudnn.h>
class BatchNorm final : public Layer{
public:
	BatchNorm(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, cudnnBatchNormMode_t bnMode, int bitchSize, const char* layerName);
	~BatchNorm() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float learningRate) override;
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t bnScaleBiasDesc_;
	cudnnTensorDescriptor_t gradDesc_;
	cudnnBatchNormMode_t bnMode_;
	size_t bitchSize_;
	__half* inData_;
	__half* outData_;
	float epsilon_;
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