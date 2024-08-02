#pragma once
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>

class LSTMLayer final : public Layer{
public:
	const bool useAdamW_ = false;
	LSTMLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int seqLength, int inC, int outC, int numLayers, const char* layerName, bool train);
	~LSTMLayer() override;
	__half* Forward(__half* data, bool train) override;
	__half* Backward(__half* grad) override;
	void UpdateParameters(float learningRate) override;
	void SaveParameters(std::ofstream& file, float* buffer) override;
	void LoadParameters(std::ifstream& file, float* buffer) override;
	void SaveOptimizerState(std::ofstream& file, float* buffer) override;
	void LoadOptimizerState(std::ifstream& file, float* buffer) override;
	bool HasParameters() override;
	bool HasOptimizerState() override;
	size_t GetParameterSize() override;
	size_t GetOptimizerStateSize() override;
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	cudnnRNNDescriptor_t rnnDesc_;
	cudnnTensorDescriptor_t* inDesc_;
	cudnnTensorDescriptor_t* outDescs_;
	cudnnTensorDescriptor_t hxDesc_;
	cudnnTensorDescriptor_t cxDesc_;
	cudnnTensorDescriptor_t hyDesc_;
	cudnnTensorDescriptor_t cyDesc_;
	cudnnFilterDescriptor_t wDesc_;
	__half* weights_;
	__half* gradWeights_;
	__half* outData_;
	__half* inData_;
	__half* hx_;
	__half* cx_;
	__half* hy_;
	__half* cy_;
	void* workspace_;
	size_t workspaceSize_;
	void* reserveSpace_;
	size_t reserveSpaceSize_;
	float* m_Weights_;
	float* v_Weights_;
	int t_ = 1;
	const int batchSize_;
	const int seqLength_;
	const int inC_;
	const int outC_;
	const int numLayers_;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
};