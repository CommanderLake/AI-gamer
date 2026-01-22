#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
class SwinBlockLayer final : public Layer{
public:
	SwinBlockLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int ffDim, int numHeads, int patchRows, int patchCols, int windowSize, int shiftSize, const char* layerName, bool train, float weightDecay, int gradAccumLength);
	~SwinBlockLayer() override;
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
private:
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	int batchSize_;
	int tokens_;
	int embedDim_;
	int ffDim_;
	int gradAccumLength_;
	std::vector<Layer*> layers_;
	const float mixFwd_ = 1.0f;
	const float mixBwd_ = 1.0f;
};
