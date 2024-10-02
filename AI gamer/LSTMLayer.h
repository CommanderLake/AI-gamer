#pragma once
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
class LSTMLayer final : public Layer{
public:
	const bool useAdamW_ = true;
	LSTMLayer(cudnnHandle_t cudnnHandle, int seqLength, int numLayers, int hiddenSize, int batchSize, int inC, const char* layerName, bool train);
	~LSTMLayer() override;
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
	cudnnDropoutDescriptor_t dropoutDesc_;
	cudnnRNNDescriptor_t rnnDesc_;
	cudnnTensorDescriptor_t* xDesc_;
	cudnnTensorDescriptor_t* yDesc_;
	cudnnTensorDescriptor_t hDesc_;
	cudnnTensorDescriptor_t cDesc_;
	cudnnFilterDescriptor_t weightDesc_;
	size_t stateSize_;
	void* dropoutStates_;
	__half* weights_;
	__half* gradWeights_;
	__half* outData_;
	__half* inData_;
	__half* gradOut_;
	__half* dx_;
	size_t weightSpaceSize_;
	void* workspace_;
	size_t workspaceSize_;
	void* reserveSpace_;
	size_t reserveSpaceSize_;
	__half *m_Weights_, *v_Weights_;
	int t_ = 1;
	const int batchSize_;
	const int seqLength_;
	int hiddenSize_;
	int inC_;
	const int numLayers_;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	float weightDecay_;
};