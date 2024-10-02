#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <cudnn.h>
class FCLayer final : public Layer{
public:
	const bool useAdamW_ = true;
	FCLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inC, int outC, const char* layerName, bool train, float weightDecay);
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
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	cudnnTensorDescriptor_t biasDesc_;
	int batchSize_;
	int inC_;
	int outC_;
	__half* weights_;
	__half* bias_;
	__half* outData_;
	__half* gradOut_;
	__half* gradWeights_;
	__half* gradBias_;
	const __half* inData_;
	__half *m_Weights_, *v_Weights_;
	__half *m_Bias_, *v_Bias_;
	int t_ = 1;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
	float weightDecay_;
};