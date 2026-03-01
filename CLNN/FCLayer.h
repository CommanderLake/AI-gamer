#pragma once
#include "Layer.h"
#include "CuCommon.cuh"
class FCLayer final : public Layer{
public:
	const bool useAdamW_ = true;
	FCLayer(int batchSize, int inC, int outC, std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod, bool useBias = false);
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
	int batchSize_, inC_, outC_;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	__half* gradWeights_ = nullptr;
	__half* biases_ = nullptr;
	__half* gradBiases_ = nullptr;
	const __half* inData_ = nullptr;
	__half *m_Weights_ = nullptr, *v_Weights_ = nullptr;
	__half *m_Biases_ = nullptr, *v_Biases_ = nullptr;
	const bool useBias_;
	int t_ = 0;
	const float alpha_ = 1.0f;
	float alphaWeights_ = 1.0f;
	const float beta0_ = 0.0f;
	const float beta1_ = 1.0f;
	float weightDecay_;
	int gradAccumLength_;
	int accumCount_ = 0;
};