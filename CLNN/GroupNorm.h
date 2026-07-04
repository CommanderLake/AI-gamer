#pragma once
#include "Layer.h"
class __declspec(dllexport) GroupNorm final : public Layer{
public:
	GroupNorm(int batchSize, int channels, int height, int width, int groups, std::string layerName, bool train, float epsilon = 1e-5f);
	~GroupNorm() override;
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
	void CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks) override;
	int batchSize_, outC_, outHW_, height_, width_, groups_, channelsPerGroup_;
	float epsilon_;
	__half* inData_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	float* gamma_ = nullptr;
	float* beta_ = nullptr;
	float* gradGamma_ = nullptr;
	float* gradBeta_ = nullptr;
	float* mean_ = nullptr;
	float* invStd_ = nullptr;
	float* mGamma_ = nullptr;
	float* vGamma_ = nullptr;
	float* mBeta_ = nullptr;
	float* vBeta_ = nullptr;
	void* workspace_ = nullptr;
	size_t workspaceSize_ = 0;
	int t_ = 1;
	bool trainingAllocated_ = false;
};
