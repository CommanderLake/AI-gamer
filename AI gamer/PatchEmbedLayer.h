#pragma once
#include "Layer.h"
#include "WeightInitMethod.h"
#include <cublas_v2.h>
#include <cudnn.h>
class PatchEmbedLayer final : public Layer{
public:
	const bool useAdamW_ = true;
	PatchEmbedLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int inC, int inH, int inW, int patchSize, int embedDim, const char* layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod);
	~PatchEmbedLayer() override;
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
	cudnnHandle_t cudnn_;
	cublasHandle_t cublas_;
	int batchSize_, inC_, inH_, inW_;
	int patchSize_, embedDim_;
	int patchRows_, patchCols_;
	int patchDim_;
	int numPatches_;
	__half* patchBuffer_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	__half* gradWeights_ = nullptr;
	__half* posEmbed_ = nullptr;
	__half* gradPosEmbed_ = nullptr;
	const __half* inData_ = nullptr;
	__half *m_Weights_ = nullptr, *v_Weights_ = nullptr;
	__half *m_PosEmbed_ = nullptr, *v_PosEmbed_ = nullptr;
	cudnnTensorDescriptor_t posDesc_{};
	int t_ = 1;
	const float alpha_ = 1.0f;
	float alphaWeights_ = 1.0f;
	const float beta0_ = 0.0f;
	const float beta1_ = 1.0f;
	float weightDecay_;
	int gradAccumLength_;
	int accumCount_ = 0;
	int posCount_ = 0;
};