#pragma once
#include "Layer.h"
#include "WeightInitMethod.h"
#include <cudnn.h>
#include <cublas_v2.h>
class LSTMLayer final : public Layer{
public:
	const bool useAdamW_ = false;
	LSTMLayer(cudnnHandle_t cudnnHandle, int seqLength, int numLayers, int hiddenSize, int batchSize, int inC, const char* layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
	~LSTMLayer() override;
	__half* Forward(__half* x) override;
	__half* Backward(__half* dy) override;
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
	cudnnTensorDescriptor_t* xDescs_;
	cudnnTensorDescriptor_t hcxyDesc_;
	cudnnRNNDataDescriptor_t xDesc_;
	cudnnRNNDataDescriptor_t yDesc_;
	cudnnFilterDescriptor_t weightDesc_;
	cudnnTensorDescriptor_t wTensDesc_;
	size_t stateSize_;
	void* dropoutStates_ = nullptr;
	__half* gradWeights_;
	__half* x_ = nullptr;
	__half* y_ = nullptr;
	__half* dx_ = nullptr;
	__half* hxy_ = nullptr;
	__half* cxy_ = nullptr;
	size_t weightSpaceSize_ = 0;
	void* workspace_ = nullptr;
	size_t workspaceSize_;
	void* reserveSpace_ = nullptr;
	size_t reserveSpaceSize_;
	__half *m_Weights_ = nullptr, *v_Weights_ = nullptr;
	__half* dGrads_ = nullptr;
	int* seqLengths_ = nullptr;
	const int numLayers_, batchSize_, seqLength_, inC_, hiddenSize_;
	int t_ = 0;
	float weightDecay_;
	float alphaWeights_ = 1.0f;
	float beta0_ = 0.0f;
	float beta1_ = 1.0f;
	int gradAccumLength_;
	int accumCount_ = 0;
};