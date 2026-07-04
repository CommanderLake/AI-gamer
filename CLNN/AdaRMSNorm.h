#pragma once
#include "Layer.h"
class __declspec(dllexport) AdaRMSNorm final : public Layer{
public:
	AdaRMSNorm(int batchSize, int channels, int height, int width, std::string layerName, bool train = false, float epsilon = 1e-6f);
	~AdaRMSNorm() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void SaveParameters(std::ofstream& file, unsigned char* buffer) override;
	void LoadParameters(std::ifstream& file, unsigned char* buffer) override;
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
	void SetTrain(bool enable) override;
	void SetModulation(const __half* scale, const __half* shift, const __half* gate, int modulationBatchSize, int rowsPerModulation = 1);
	void ClearModulation();
	int batchSize_, outC_, outHW_, height_, width_;
	float epsilon_;
	__half* outData_ = nullptr;
	float* gamma_ = nullptr;
	float* invRms_ = nullptr;
	const __half* scale_ = nullptr;
	const __half* shift_ = nullptr;
	const __half* gate_ = nullptr;
	int modulationBatchSize_ = 0;
	int rowsPerModulation_ = 1;
};
