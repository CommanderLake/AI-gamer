#pragma once
#include "Layer.h"
class __declspec(dllexport) TimestepEmbedding final : public Layer{
public:
	TimestepEmbedding(int batchSize, int embeddingDim, std::string layerName, float maxPeriod = 10000.0f);
	~TimestepEmbedding() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SetTimestepsDevice(const float* timesteps);
	void SetTimestepsHost(const float* timesteps);
	void ClearTimesteps();
	float* GetTimestepGrad();
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
private:
	int batchSize_;
	int embeddingDim_;
	float maxPeriod_;
	__half* outData_ = nullptr;
	__half* gradHalfTimesteps_ = nullptr;
	float* ownedTimesteps_ = nullptr;
	float* gradTimesteps_ = nullptr;
	const float* timesteps_ = nullptr;
	const float* lastFloatTimesteps_ = nullptr;
	const __half* lastHalfTimesteps_ = nullptr;
	bool lastForwardUsedHalf_ = false;
};
