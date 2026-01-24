#pragma once
#include "Layer.h"
#include <cublas_v2.h>
#include <string>
#include <vector>
class SwinUnetLayer final : public Layer{
public:
	SwinUnetLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int patchRows, int patchCols, int embedDim, int ffDim, int numHeads, int windowSize, int shiftStride, int depth0, int depth1, int depth2, int depth3, const char* layerName, bool train, float weightDecay, int gradAccumLength);
	~SwinUnetLayer() override;
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
	int patchRows_;
	int patchCols_;
	int embedDim_;
	int ffDim_;
	int numHeads_;
	int windowSize_;
	int shiftStride_;
	int depth0_;
	int depth1_;
	int depth2_;
	int depth3_;
	int tokens_;
	int mergedRows_;
	int mergedCols_;
	int mergedTokens_;
	int reducedRows_;
	int reducedCols_;
	int reducedTokens_;
	std::vector<std::string> layerNames_;
	std::vector<Layer*> enc0_;
	std::vector<Layer*> enc1_;
	std::vector<Layer*> enc2_;
	std::vector<Layer*> dec1_;
	std::vector<Layer*> dec0_;
	Layer* mergeProj1_ = nullptr;
	Layer* mergeProj2_ = nullptr;
	Layer* expandProj2_ = nullptr;
	Layer* expandProj1_ = nullptr;
	__half* skipBuffer0_ = nullptr;
	__half* skipBuffer1_ = nullptr;
	__half* mergePacked1_ = nullptr;
	__half* mergePacked1Grad_ = nullptr;
	__half* mergePacked2_ = nullptr;
	__half* mergePacked2Grad_ = nullptr;
	__half* expandPacked1Grad_ = nullptr;
	__half* expandPacked2Grad_ = nullptr;
	__half* expandedTokens1_ = nullptr;
	__half* expandedTokens2_ = nullptr;
};
