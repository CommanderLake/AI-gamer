#pragma once
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
class MultiHeadAttentionLayer final : public Layer{
public:
	const bool useAdamW_ = true;
	MultiHeadAttentionLayer(cudnnHandle_t cudnnHandle, int batchSize, int timeSize, int vectorSize, int numHeads, const char* layerName, bool train, float weightDecay);
	~MultiHeadAttentionLayer() override;
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
	cudnnHandle_t cudnnHandle_;
	cudnnAttnDescriptor_t attnDesc_;
	cudnnSeqDataDescriptor_t qkvDesc_, outDesc_;
	int batchSize_, timeSize_, vectorSize_;
	int numHeads_;
	__half *inData_, *outData_;
	__half *gradWeights_, *gradKeys_, *gradValues_, *gradOut_;
	size_t workspaceSize_;
	void* workspace_;
	size_t reserveSpaceSize_;
	void* reserveSpace_;
	__half *m_Weights_, *v_Weights_;
	int t_ = 1;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	size_t weightSize_;
	int* d_SeqLengths;
	std::vector<int>* loWinIdx;
	std::vector<int>* hiWinIdx;
	float weightDecay_;
};