#pragma once
#include "Layer.h"
#include "CuCommon.cuh"
#include <unordered_map>
#include <vector>
class GELULayer;
class PatchEmbedLayer;
class PatchMergingLayer;
class PatchExpandingLayer;
class SwinBlockLayer;
class LayerNorm;
class SwinUnetLayer final : public Layer{
public:
	SwinUnetLayer(cudnnHandle_t cudnnHandle, int batchSize, int inHeight, int inWidth, int patchSize, int embedH, int embedW, int blocksPerStage, int numStages, int baseHeads, int baseWindowSize, float maxDropPathRate, std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
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
	void CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks) override;
private:
	struct SwinBlockWorkspace{
		size_t windowBytes = 0;
		size_t tokenBytes = 0;
		__half* windowedInput = nullptr;
		__half* windowedGrad = nullptr;
		__half* tokens = nullptr;
	};
	struct AttentionWorkspace{
		size_t workspaceBytes = 0;
		size_t packedBytes = 0;
		size_t gradWorkspaceBytes = 0;
		__half* workspace = nullptr;
		__half* qPacked = nullptr;
		__half* kPacked = nullptr;
		__half* vPacked = nullptr;
		__half* attnOutPacked = nullptr;
		__half* dQPacked = nullptr;
		__half* dKPacked = nullptr;
		__half* dVPacked = nullptr;
		float* gradWorkspace = nullptr;
	};
	struct SkipConnection{
		int elements = 0;
		size_t bytes = 0;
		__half* scratch = nullptr;
		bool isActivation = false;
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
	int batchSize_;
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
	LayerNorm* postNorm_ = nullptr;
	GELULayer* postGELU_ = nullptr;
	std::vector<EncoderStage> encoderStages_;
	std::vector<DecoderStage> decoderStages_;
	std::unordered_map<unsigned long long, float*> attentionMaskCache_;
	SwinBlockWorkspace blockWorkspace_;
	AttentionWorkspace attentionWorkspace_;
};