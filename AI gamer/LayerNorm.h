#pragma once
#include "Layer.h"
#include <cudnn.h>
class LayerNorm final : public Layer{
public:
	LayerNorm(int batchSize, int channels, int height, int width, const char* layerName, float weightDecay);
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
	int ogbs_;
	int batchSize_;
	int outC_;
	int outHW_;
	__half* inData_;
	__half* outData_;
	float* gamma_, *beta_;
	float* gradGamma_, *gradBeta_;
	float* mean_, *variance_;
	float *mGamma_, *vGamma_;
	float *mBeta_, *vBeta_;
	int t_ = 1;
	float weightDecay_;
};