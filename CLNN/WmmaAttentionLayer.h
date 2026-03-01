#pragma once
#include "Layer.h"
#include "CuCommon.cuh"
#include <vector>
class WmmaAttentionLayer final : public Layer{
public:
	WmmaAttentionLayer(cudnnHandle_t cudnnHandle, int batchSize, int tokens, int embedDim, int numHeads, std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
	WmmaAttentionLayer(cudnnHandle_t cudnnHandle, int batchSize, int tokens, int embedDim, int numHeads, std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod,
		__half* sharedWorkspace, __half* sharedQPacked, __half* sharedKPacked, __half* sharedVPacked, __half* sharedAttnOutPacked, __half* sharedDQPacked, __half* sharedDKPacked, __half* sharedDVPacked, float* sharedAttnGradWorkspace);
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
	void SetAttentionMask(const float* attentionMask, int maskBatchSize, int maskHeads);
	void InitRelativePositionBias(int windowHeight, int windowWidth, const std::vector<int>& relPosIndex);
	cudnnHandle_t cudnnHandle_;
	int batchSize_, tokens_, embedDim_, numHeads_;
	int headDim_;
	__half* outData_ = nullptr;
	const __half* inData_ = nullptr;
	__half* outGrad_ = nullptr;
	__half *qkvWeightsBase_ = nullptr;
	__half *qWeights_ = nullptr, *kWeights_ = nullptr, *vWeights_ = nullptr, *oWeights_ = nullptr;
	__half *gradQkvBase_ = nullptr;
	__half *gradQ_ = nullptr, *gradK_ = nullptr, *gradV_ = nullptr, *gradOut_ = nullptr;
	__half *m_Q_ = nullptr, *v_Q_ = nullptr, *m_K_ = nullptr, *v_K_ = nullptr, *m_V_ = nullptr, *v_V_ = nullptr, *m_O_ = nullptr, *v_O_ = nullptr;
	__half *dQ = nullptr, *dK = nullptr, *dV = nullptr;
	__half *qPacked_ = nullptr, *kPacked_ = nullptr, *vPacked_ = nullptr, *attnOutPacked_ = nullptr;
	__half *dQPacked_ = nullptr, *dKPacked_ = nullptr, *dVPacked_ = nullptr;
	__half* workspace_;
	float* attnGradWorkspace_ = nullptr;
	size_t attnGradWorkspaceSize_ = 0;
	const float* attentionMask_ = nullptr;
	int maskBatchSize_ = 0;
	int maskHeads_ = 0;
	float* relPosBias_ = nullptr;
	float* gradRelPosBias_ = nullptr;
	float* m_relPosBias_ = nullptr;
	float* v_relPosBias_ = nullptr;
	int* relPosIndex_ = nullptr;
	int relPosSize_ = 0;
	bool useRelPosBias_ = false;
	int gradAccumLength_;
	int t_ = 1;
	const float zero_ = 0.0f;
	const float one_ = 1.0f;
	float weightDecay_;
	float alphaWeights_ = 1.0f;
	int accumCount_ = 0;
	bool ownsTemporaries_ = true;
	bool ownsPackedGradTemporaries_ = true;
	bool ownsAttnGradWorkspace_ = true;
};
