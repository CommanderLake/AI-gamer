#pragma once
#include "Layer.h"
#include <cudnn.h>

class LayerNorm final : public Layer{
public:
	LayerNorm(cudnnTensorDescriptor_t outDesc, int batchSize, const char* layerName);
	~LayerNorm() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float learningRate) override;

private:
	cudnnTensorDescriptor_t gradDesc_;
	size_t batchSize_;
	int outC_;
	int outHW_;
	size_t outCHW_;
	__half* inData_;
	__half* outData_;
	float epsilon_;
	float* gamma_;
	float* beta_;
	float* gradGamma_;
	float* gradBeta_;
	float* mean_;
	float* variance_;
	float *mGamma_, *vGamma_;
	float *mBeta_, *vBeta_;
	int t_ = 1;
};