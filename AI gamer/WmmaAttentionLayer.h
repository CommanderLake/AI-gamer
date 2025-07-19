#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <cudnn.h>
// A lightweight attention layer operating on patch tokens.
// Q, K and V projections are computed with fully connected layers and
// the attention scores are obtained using a WMMA accelerated kernel.
class WmmaAttentionLayer final : public Layer{
public:
	WmmaAttentionLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int numHeads, const char* layerName, bool train, float weightDecay);
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
private:
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	int batchSize_, tokens_, embedDim_, numHeads_;
	int headDim_;
	__half* outData_ = nullptr;
	const __half* inData_ = nullptr;
	__half* outGrad_ = nullptr;
	__half *qWeights_, *kWeights_, *vWeights_, *oWeights_;
	__half *gradQ_, *gradK_, *gradV_, *gradOut_;
	__half *m_Q_, *v_Q_, *m_K_, *v_K_, *m_V_, *v_V_, *m_O_, *v_O_;
	__half *dQ, *dK, *dV;
	__half* workspace_;
	int t_ = 1;
	const float alpha_ = 1.0f;
	const float beta0_ = 0.0f;
	const float beta1_ = 1.0f;
	float weightDecay_;
};