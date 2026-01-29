#pragma once
#include "Layer.h"
#include "WeightInitMethod.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
class LayerNorm;
class WmmaAttentionLayer;
class FCLayer;
class GELULayer;
class Dropout;
class DropPath;
class SwinBlockLayer final : public Layer{
public:
	SwinBlockLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int nTokens, int embedDim, int ffDim, int numHeads, int patchRows, int patchCols, int windowHeight, int windowWidth, int shiftHeight, int shiftWidth, float dropPathRate,
		std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod, __half* windowedInput = nullptr, __half* windowedGrad = nullptr, __half* tokens = nullptr, __half* residualGrad = nullptr);
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
	int nTokens_;
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
	std::vector<Layer*> layers_;
	LayerNorm* norm1_;
	WmmaAttentionLayer* attention_;
	Dropout* attnDrop_;
	DropPath* attnDropPath_;
	LayerNorm* norm2_;
	FCLayer* fc1_;
	GELULayer* gelu_;
	FCLayer* fc2_;
	Dropout* ffDrop_;
	DropPath* ffDropPath_;
	__half* windowedInput_ = nullptr;
	__half* windowedGrad_ = nullptr;
	__half* tokens_ = nullptr;
	__half* residualGrad_ = nullptr;
	float* attentionMask_ = nullptr;
	bool ownsWorkspace_ = true;
	cudnnTensorDescriptor_t outDesc_;
	const float mixFwd_ = 1.0f;
	const float mixBwd_ = 1.0f;
};
