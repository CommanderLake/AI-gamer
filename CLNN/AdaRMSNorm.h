#pragma once
#include "Layer.h"
class __declspec(dllexport) AdaRMSNorm final : public Layer{
public:
	AdaRMSNorm(int batchSize, int channels, int height, int width, std::string layerName, bool train = false, float epsilon = 1e-6f);
	~AdaRMSNorm() override;
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
	void SetModulation(const __half* scale, const __half* shift, const __half* gate, int modulationBatchSize, int rowsPerModulation = 1);
	void SetModulationGradients(__half* gradScale, __half* gradShift, __half* gradGate);
	void ClearModulation();
	__half* GetScaleGrad();
	__half* GetShiftGrad();
	__half* GetGateGrad();
	int batchSize_, outC_, outHW_, height_, width_;
	float epsilon_;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	float* gamma_ = nullptr;
	float* gradGamma_ = nullptr;
	float* mGamma_ = nullptr;
	float* vGamma_ = nullptr;
	float* invRms_ = nullptr;
	void* workspace_ = nullptr;
	size_t workspaceSize_ = 0;
	const __half* inData_ = nullptr;
	const __half* scale_ = nullptr;
	const __half* shift_ = nullptr;
	const __half* gate_ = nullptr;
	__half* gradScale_ = nullptr;
	__half* gradShift_ = nullptr;
	__half* gradGate_ = nullptr;
	int modulationBatchSize_ = 0;
	int rowsPerModulation_ = 1;
	int t_ = 1;
	bool trainingAllocated_ = false;
};
