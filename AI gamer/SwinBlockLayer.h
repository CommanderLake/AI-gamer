#pragma once
#include "Layer.h"
#include "WeightInitMethod.h"
#include <cublas_v2.h>
#include <cudnn.h>

class LayerNorm;
class WmmaAttentionLayer;
class FCLayer;
class GELULayer;
class Dropout;

class SwinBlockLayer final : public Layer{
public:
	SwinBlockLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int ffDim, int numHeads, int patchRows, int patchCols, int windowHeight, int windowWidth, int shiftHeight, int shiftWidth, const char* layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
	~SwinBlockLayer() override;
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
	int batchSize_;
	int tokens_;
	int embedDim_;
	int ffDim_;
	int numHeads_;
	int patchRows_;
	int patchCols_;
	int windowHeight_;
	int windowWidth_;
	int shiftHeight_;
	int shiftWidth_;
	int windowTokens_;
	int windowCount_;
	int windowBatch_;
	LayerNorm* norm1_;
	WmmaAttentionLayer* attention_;
	Dropout* attnDrop_;
	LayerNorm* norm2_;
	FCLayer* fc1_;
	GELULayer* gelu_;
	FCLayer* fc2_;
	Dropout* ffDrop_;
	__half* windowedInput_ = nullptr;
	__half* windowedGrad_ = nullptr;
	__half* tokenBuffer_ = nullptr;
	cudnnTensorDescriptor_t outDesc_;
	const float mixFwd_ = 1.0f;
	const float mixBwd_ = 1.0f;
};
