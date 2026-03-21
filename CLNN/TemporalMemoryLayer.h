#pragma once
#include "Layer.h"
#include "CuCommon.cuh"
class TemporalMemoryLayer final : public Layer{
public:
	TemporalMemoryLayer(int batchSize, int embedDim, int numHeads, int maxContext, std::string layerName, bool train, float weightDecay, int gradAccumLength);
	~TemporalMemoryLayer() override;
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
	void CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks) override;
	void SetMemory(const __half* memory, const int* validCounts);
private:
	int batchSize_;
	int embedDim_;
	int numHeads_;
	int headDim_;
	int maxContext_;
	float weightDecay_;
	int gradAccumLength_;
	int accumCount_ = 0;
	int t_ = 1;
	float alphaWeights_ = 1.0f;
	const float zero_ = 0.0f;
	const float one_ = 1.0f;
	const __half* memory_ = nullptr;
	const int* validCounts_ = nullptr;
	const __half* inData_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	__half* qWeights_ = nullptr;
	__half* kWeights_ = nullptr;
	__half* vWeights_ = nullptr;
	__half* oWeights_ = nullptr;
	__half* gradQ_ = nullptr;
	__half* gradK_ = nullptr;
	__half* gradV_ = nullptr;
	__half* gradO_ = nullptr;
	__half* m_Q_ = nullptr;
	__half* v_Q_ = nullptr;
	__half* m_K_ = nullptr;
	__half* v_K_ = nullptr;
	__half* m_V_ = nullptr;
	__half* v_V_ = nullptr;
	__half* m_O_ = nullptr;
	__half* v_O_ = nullptr;
	__half* qProj_ = nullptr;
	__half* kProj_ = nullptr;
	__half* vProj_ = nullptr;
	__half* context_ = nullptr;
	__half* qPacked_ = nullptr;
	__half* kPacked_ = nullptr;
	__half* vPacked_ = nullptr;
	__half* contextPacked_ = nullptr;
	__half* gradQPacked_ = nullptr;
	__half* gradKPacked_ = nullptr;
	__half* gradVPacked_ = nullptr;
	float* attnWeights_ = nullptr;
	float* gradScores_ = nullptr;
};
