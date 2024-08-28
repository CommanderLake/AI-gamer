#pragma once
#include "Layer.h"
#include <cudnn.h>
class BatchNorm final : public Layer{
public:
	const bool useAdamW_ = true;
	BatchNorm(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, cudnnBatchNormMode_t bnMode, int bitchSize, const char* layerName, bool train, float weightDecay);
	~BatchNorm() override;
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
	float *m_BnScale_, *v_BnScale_;
	float *m_BnBias_, *v_BnBias_;
	int t_ = 1;
	int outC_;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
	float weightDecay_;
};