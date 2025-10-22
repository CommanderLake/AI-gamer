#pragma once
#include "Layer.h"
class LayerNorm final : public Layer{
public:
	LayerNorm(int batchSize, int channels, int height, int width, int tokensPerSample, const char* layerName, bool train);
	~LayerNorm() override;
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
	int batchSize_, tokenBatchSize_, outC_;
	int outHW_, height_, width_;
	__half* inData_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	float *gamma_, *beta_;
	float *gradGamma_, *gradBeta_;
	float *mean_, *variance_;
	float *mGamma_, *vGamma_;
	float *mBeta_, *vBeta_;
	float *workspace_;
	int workspaceSize_;
	int t_ = 1;
};