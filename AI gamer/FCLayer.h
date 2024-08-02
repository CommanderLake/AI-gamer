#pragma once
#include "Layer.h"
#include <cublas_v2.h>

class FCLayer final : public Layer{
public:
	const bool useAdamW_ = false;
	FCLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inputSize, int outputSize, const char* layerName, bool train);
	~FCLayer() override;
	__half* Forward(__half* data, bool train) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float learningRate) override;
	void SaveParameters(std::ofstream& file, float* buffer) override;
	void LoadParameters(std::ifstream& file, float* buffer) override;
	void SaveOptimizerState(std::ofstream& file, float* buffer) override;
	void LoadOptimizerState(std::ifstream& file, float* buffer) override;
	bool HasParameters() override;
	bool HasOptimizerState() override;
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
	__half* gradIn_;
	__half* gradWeights_;
	__half* gradBias_;
	const __half* inData_;
	float* m_Weights_;
	float* v_Weights_;
	float* m_Bias_;
	float* v_Bias_;
	int t_ = 1;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
};