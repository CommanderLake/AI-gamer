#pragma once
#include "Layer.h"
#include <cudnn.h>
class SpatialAttentionLayer : public Layer{
public:
	bool useAdamW_ = true;
	SpatialAttentionLayer(cudnnHandle_t cudnnHandle, int attentionChannels, int numHeads, int batchSize, int channels, int height, int width, const char* layerName, bool train, float weightDecay);
	~SpatialAttentionLayer() override;
	__half* Forward(__half* data) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float learningRate) override;
	void SaveParameters(std::ofstream& file, unsigned char* buffer) override;
	void LoadParameters(std::ifstream& file, unsigned char* buffer) override;
	void SaveOptimizerState(std::ofstream& file, unsigned char* buffer) override;
	void LoadOptimizerState(std::ifstream& file, unsigned char* buffer) override;
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t attentionDesc_;
	cudnnFilterDescriptor_t keyQueryFilterDesc_, valueFilterDesc_;
	cudnnConvolutionDescriptor_t keyQueryConvDesc_, valueConvDesc_;
	ConvolutionAlgorithms keyQueryAlgos_, valueAlgos_;
	int batchSize_, inC_, inH_, inW_;
	int dwWeightSizeAll_, pwWeightSizeAll_;
	int dwWeightSizeSingle_, pwWeightSizeSingle_;
	int numHeads_;
	int kqHeadSize_, vHeadSize_;
	int channelsPerHead_, attC_, attCPerHead_;
	__half *inData_, *outData_;
	__half *keyMap_, *queryMap_, *valueMap_, *attentionScores_;
	__half *gradOut_;
	__half *keyWeights_, *queryWeights_, *valueWeights_;
	__half *gradKeyWeights_, *gradQueryWeights_, *gradValueWeights_;
	__half *m_Key_, *v_Key_, *m_Query_, *v_Query_, *m_Pointwise_, *v_Pointwise_;
	__half *gradValueMap_, *gradAttention_, *gradKeyMap_, *gradQueryMap_;
	__half* headOutput_;
	int t_ = 1;
	void *keyQueryWorkspace_, *valueWorkspace_;
	const float alpha = 1.0f;
	const float alpha05 = 0.5f;
	const float beta0 = 0.0f;
	const float beta1 = 1.0f;
	float weightDecay_;
};