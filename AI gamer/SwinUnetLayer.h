#pragma once
#include "Layer.h"
#include "WeightInitMethod.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
class PatchEmbedLayer;
class PatchMergingLayer;
class PatchExpandingLayer;
class SwinBlockLayer;
class LayerNorm;
class SwinUnetLayer final : public Layer{
public:
	SwinUnetLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inChannels, int inHeight, int inWidth, int patchSize, int embedH, int embedW, int blocksPerStage, int numStages, int baseHeads, int baseWindowSize, float maxDropPathRate, const char* layerName, bool train,
				float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
	~SwinUnetLayer() override;
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
	struct SkipConnection{
		int elements = 0;
		size_t bytes = 0;
		__half* data = nullptr;
		__half* grad = nullptr;
	};
	struct EncoderStage{
		std::vector<SwinBlockLayer*> blocks;
		PatchMergingLayer* merge = nullptr;
		SkipConnection skip;
	};
	struct DecoderStage{
		PatchExpandingLayer* expand = nullptr;
		std::vector<SwinBlockLayer*> blocks;
	};
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	int batchSize_;
	int inChannels_;
	int inHeight_;
	int inWidth_;
	int patchSize_;
	int embedH_;
	int embedW_;
	int blocksPerStage_;
	int numStages_;
	int baseHeads_;
	int baseWindowSize_;
	float maxDropPathRate_;
	float weightDecay_;
	int gradAccumLength_;
	WeightInitMethod weightInitMethod_;
	PatchEmbedLayer* patchEmbed_ = nullptr;
	LayerNorm* postNorm_ = nullptr;
	std::vector<EncoderStage> encoderStages_;
	std::vector<DecoderStage> decoderStages_;
};