#pragma once
#include "Layer.h"
class __declspec(dllexport) RMSNorm final : public Layer{
public:
	RMSNorm(int batchSize, int channels, int height, int width, std::string layerName, bool train, bool spatialMode = false, float epsilon = 1e-6f);
	~RMSNorm() override;
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
	int batchSize_, outC_, outHW_, height_, width_;
	int normSize_;
	bool spatialMode_;
	float epsilon_;
	__half* inData_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	float* gamma_ = nullptr;
	float* gradGamma_ = nullptr;
	float* invRms_ = nullptr;
	float* mGamma_ = nullptr;
	float* vGamma_ = nullptr;
	void* workspace_ = nullptr;
	size_t workspaceSize_ = 0;
	int t_ = 1;
	bool trainingAllocated_ = false;
};
