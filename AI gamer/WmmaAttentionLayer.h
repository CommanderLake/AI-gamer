#pragma once
#include "Layer.h"
#include "WeightInitMethod.h"
#include <cublas_v2.h>
#include <cudnn.h>
class WmmaAttentionLayer final : public Layer{
public:
	WmmaAttentionLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int numHeads, const char* layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
	~WmmaAttentionLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float lr) override;
	void SaveParameters(std::ofstream& file, unsigned char* buffer) override;
	void LoadParameters(std::ifstream& file, unsigned char* buffer) override;
	void SaveOptimizerState(std::ofstream& file, unsigned char* buffer) override;
	void LoadOptimizerState(std::ifstream& file, unsigned char* buffer) override;
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
	void SetTrain(bool enable) override;
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	int batchSize_, tokens_, embedDim_, numHeads_;
	int headDim_;
	__half* outData_ = nullptr;
	const __half* inData_ = nullptr;
	__half* outGrad_ = nullptr;
	__half *qkvWeightsBase_ = nullptr;
	__half *qWeights_ = nullptr, *kWeights_ = nullptr, *vWeights_ = nullptr, *oWeights_ = nullptr;
	__half *gradQkvBase_ = nullptr;
	__half *gradQ_ = nullptr, *gradK_ = nullptr, *gradV_ = nullptr, *gradOut_ = nullptr;
	__half *m_Q_, *v_Q_, *m_K_, *v_K_, *m_V_, *v_V_, *m_O_, *v_O_;
	__half *dQ = nullptr, *dK = nullptr, *dV = nullptr;
	__half *qPacked_ = nullptr, *kPacked_ = nullptr, *vPacked_ = nullptr, *attnOutPacked_ = nullptr;
	__half *dQPacked_ = nullptr, *dKPacked_ = nullptr, *dVPacked_ = nullptr;
	__half* workspace_;
	float* attnGradWorkspace_ = nullptr;
	size_t attnGradWorkspaceSize_ = 0;
	int gradAccumLength_;
	int t_ = 1;
	const float zero_ = 0.0f;
	const float one_ = 1.0f;
	float weightDecay_;
	float alphaWeights_ = 1.0f;
	int accumCount_ = 0;
};