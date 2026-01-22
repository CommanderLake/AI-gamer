#pragma once
#include "Layer.h"
#include "WeightInitMethod.h"
#include <cublas_v2.h>
#include <cudnn.h>
class WmmaAttentionLayer;
class WindowAttentionLayer final : public Layer{
public:
	WindowAttentionLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int tokens, int embedDim, int numHeads, int patchRows, int patchCols, int windowSize, int shiftSize, const char* layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
	~WindowAttentionLayer() override;
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
	int numHeads_;
	int patchRows_;
	int patchCols_;
	int windowSize_;
	int shiftSize_;
	int windowTokens_;
	int numWindows_;
	int windowBatch_;
	int gradAccumLength_;
	float weightDecay_;
	float alphaBias_;
	int biasAccumCount_ = 0;
	WmmaAttentionLayer* attention_ = nullptr;
	__half* shiftedData_ = nullptr;
	__half* windowData_ = nullptr;
	__half* mergedData_ = nullptr;
	__half* outData_ = nullptr;
	__half* shiftedGrad_ = nullptr;
	__half* windowGrad_ = nullptr;
	__half* outGrad_ = nullptr;
	float* attnMask_ = nullptr;
	float* relPosBias_ = nullptr;
	float* gradRelPosBias_ = nullptr;
	float* mRelPosBias_ = nullptr;
	float* vRelPosBias_ = nullptr;
	int* relPosIndex_ = nullptr;
	int relPosBiasSize_ = 0;
};
