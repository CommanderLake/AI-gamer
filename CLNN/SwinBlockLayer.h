#pragma once
#include "Layer.h"
#include "CuCommon.cuh"
#include <vector>
class LayerNorm;
class WmmaAttentionLayer;
class FCLayer;
class GELULayer;
class Dropout;
class DropPath;
class SwinBlockLayer final : public Layer{
public:
	SwinBlockLayer(int batchSize, int nTokens, int embedDim, int ffDim, int numHeads, int patchRows, int patchCols, int windowHeight, int windowWidth, int shiftHeight, int shiftWidth, float dropPathRate, std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod, __half* windowedInput = nullptr, __half* windowedGrad = nullptr, __half* tokens = nullptr, float* sharedAttentionMask = nullptr, bool ownsAttentionMask = true, __half* attentionWorkspace = nullptr, __half* qPacked = nullptr, __half* kPacked = nullptr, __half* vPacked = nullptr, __half* attnOutPacked = nullptr, __half* dQPacked = nullptr, __half* dKPacked = nullptr, __half* dVPacked = nullptr, float* attnGradWorkspace = nullptr);
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
	void CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks) override;
private:
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
	float* attentionMask_ = nullptr;
	bool ownsWorkspace_ = true;
	bool ownsAttentionMask_ = true;
	const float mixFwd_ = 1.0f;
	const float mixBwd_ = 1.0f;
};
