#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <cudnn.h>
class FCLayer final : public Layer{
public:
	const bool useAdamW_ = true;
	FCLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inC, int outC, const char* layerName, bool train, float weightDecay, int gradAccumLength);
	~FCLayer() override;
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
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	int ogbs_, batchSize_, inC_, outC_;
	__half* outData_ = nullptr;
	__half* gradOut_ = nullptr;
	__half* gradWeights_ = nullptr;
	const __half* inData_;
	__half *m_Weights_, *v_Weights_;
	int t_ = 0;
	const float alpha_ = 1.0f;
	float alphaWeights_ = 1.0f;
	const float beta0_ = 0.0f;
	const float beta1_ = 1.0f;
	float weightDecay_;
	int gradAccumLength_;
	int accumCount_ = 0;
};