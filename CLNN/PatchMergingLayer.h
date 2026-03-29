#pragma once
#include "Layer.h"
#include "CuCommon.cuh"
class LayerNorm;
class FCLayer;
class PatchMergingLayer final : public Layer{
public:
	PatchMergingLayer(int batchSize, int tokens, int embedDim, int patchRows, int patchCols, std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
	~PatchMergingLayer() override;
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
	int batchSize_;
	int tokens_;
	int embedDim_;
	int patchRows_;
	int patchCols_;
	int outTokens_;
	int outEmbedDim_;
	LayerNorm* norm_;
	FCLayer* reduction_;
	__half* mergedData_ = nullptr;
	__half* mergedGrad_ = nullptr;
	__half* tokenGrad_ = nullptr;
};