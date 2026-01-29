#pragma once
#include "Layer.h"
#include "WeightInitMethod.h"
#include <cublas_v2.h>
#include <cudnn.h>

class LayerNorm;
class FCLayer;

class PatchExpandingLayer final : public Layer{
public:
	PatchExpandingLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int patchRows, int patchCols, const char* layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
	~PatchExpandingLayer() override;
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
	int patchRows_;
	int patchCols_;
	int outTokens_;
	int outEmbedDim_;
	int outPatchRows_;
	int outPatchCols_;
	LayerNorm* norm_;
	FCLayer* expansion_;
	__half* expandedData_ = nullptr;
	__half* expandedTokens_ = nullptr;
	__half* mergedGrad_ = nullptr;
};