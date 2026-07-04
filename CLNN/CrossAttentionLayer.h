#pragma once
#include "Layer.h"
#include "CuCommon.h"
class __declspec(dllexport) CrossAttentionLayer final : public Layer{
public:
	CrossAttentionLayer(int batchSize, int queryTokens, int contextTokens, int embedDim, int numHeads, std::string layerName, WeightInitMethod weightInitMethod = Xavier, int contextDim = 0);
	CrossAttentionLayer(int batchSize, int queryTokens, int contextTokens, int embedDim, int numHeads, std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod = Xavier, int contextDim = 0);
	~CrossAttentionLayer() override;
	__half* Forward(__half* data) override;
	__half* Forward(__half* queryData, const __half* contextData);
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
	void SetContext(const __half* contextData);
	void ClearContext();
	void SetAttentionMask(const float* attentionMask, int maskBatchSize, int maskHeads);
	float* GetAttentionWeights() const;
	__half* GetContextGrad() const;
	int batchSize_, queryTokens_, contextTokens_, embedDim_, contextDim_, numHeads_, headDim_;
private:
	void AllocateTrainingBuffers();
	__half *qWeights_ = nullptr, *kWeights_ = nullptr, *vWeights_ = nullptr, *oWeights_ = nullptr;
	__half *gradQWeights_ = nullptr, *gradKWeights_ = nullptr, *gradVWeights_ = nullptr, *gradOWeights_ = nullptr;
	__half *m_Q_ = nullptr, *v_Q_ = nullptr, *m_K_ = nullptr, *v_K_ = nullptr, *m_V_ = nullptr, *v_V_ = nullptr, *m_O_ = nullptr, *v_O_ = nullptr;
	__half *q_ = nullptr, *k_ = nullptr, *v_ = nullptr, *attnOut_ = nullptr;
	__half *dQ_ = nullptr, *dK_ = nullptr, *dV_ = nullptr, *dAttnOut_ = nullptr;
	__half *qPacked_ = nullptr, *kPacked_ = nullptr, *vPacked_ = nullptr, *attnOutPacked_ = nullptr;
	__half *dQPacked_ = nullptr, *dKPacked_ = nullptr, *dVPacked_ = nullptr, *dAttnOutPacked_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	__half* contextGrad_ = nullptr;
	float* attentionWeights_ = nullptr;
	float* attentionGrad_ = nullptr;
	const __half* inData_ = nullptr;
	const __half* contextInData_ = nullptr;
	const __half* contextData_ = nullptr;
	const float* attentionMask_ = nullptr;
	int maskBatchSize_ = 0;
	int maskHeads_ = 0;
	size_t queryElements_ = 0;
	size_t contextElements_ = 0;
	size_t contextInputElements_ = 0;
	size_t squareProjectionElements_ = 0;
	size_t contextProjectionElements_ = 0;
	const float zero_ = 0.0f;
	const float one_ = 1.0f;
	float alphaWeights_ = 1.0f;
	float weightDecay_ = 0.0f;
	int gradAccumLength_ = 1;
	int accumCount_ = 0;
	int t_ = 1;
	bool trainingAllocated_ = false;
};
