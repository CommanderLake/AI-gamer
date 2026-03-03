#pragma once
#include "Layer.h"
#include <cudnn.h>
class BatchNorm final : public Layer{
public:
	const bool useAdamW_ = true;
	BatchNorm(cudnnHandle_t cudnnHandle, cudnnBatchNormMode_t bnMode, int batchSize, int channels, int height, int width, std::string layerName, bool train, int gradAccumLength);
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
	void SetTrain(bool enable) override;
	void CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks) override;
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t bnScaleBiasDesc_;
	cudnnBatchNormMode_t bnMode_;
	int batchSize_, outC_, outHeight_, outWidth_;
	__half* inData_ = nullptr;
	__half* outData_ = nullptr;
	float epsilon_;
	float* bnScale_ = nullptr;
	float* bnBias_ = nullptr;
	float* gradBnScale_ = nullptr;
	float* gradBnBias_ = nullptr;
	float* bnRunningMeanTrain_ = nullptr;
	float* bnRunningVarTrain_ = nullptr;
	float* bnRunningMeanInfer_ = nullptr;
	float* bnRunningVarInfer_ = nullptr;
	float* bnSavedMean_ = nullptr;
	float* bnSavedInvVariance_ = nullptr;
	float *m_BnScale_ = nullptr, *v_BnScale_ = nullptr;
	float *m_BnBias_ = nullptr, *v_BnBias_ = nullptr;
	int t_ = 0;
	const float alpha_ = 1.0f;
	float alphaWeights_ = 1.0f;
	const float beta0_ = 0.0f;
	const float beta1_ = 1.0f;
	int gradAccumLength_;
	int accumCount_ = 0;
};