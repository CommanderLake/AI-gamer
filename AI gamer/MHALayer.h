#pragma once
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
class MultiHeadAttentionLayer final : public Layer{
public:
	const bool useAdamW_ = true;
	MultiHeadAttentionLayer(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int batchSize, int seqLength, int embedSize, int numHeads, const char* layerName, bool train, float weightDecay);
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
	cudnnHandle_t cudnnHandle_;
	cublasHandle_t cublasHandle_;
	cudnnAttnDescriptor_t attnDesc_;
	cudnnSeqDataDescriptor_t qDesc_, kDesc_, vDesc_, oDesc_;
	int batchSize_;
	int seqLength_;
	int embedSize_;
	int numHeads_;
	__half *qData_, *kData_, *vData_, *oData_;
	__half *wq_, *wk_, *wv_, *wo_;
	__half *gradWq_, *gradWk_, *gradWv_, *gradWo_;
	__half* gradIn_;
	size_t workspaceSize_;
	void* workspace_;
	size_t reserveSpaceSize_;
	void* reserveSpace_;
	__half *m_Wq_, *v_Wq_, *m_Wk_, *v_Wk_, *m_Wv_, *v_Wv_, *m_Wo_, *v_Wo_;
	int t_ = 1;
	const float alpha = 1.0f;
	const float beta0 = 0.0f;
	size_t weightSize_;
	int* d_SeqLengthsQo;
	int* d_SeqLengthsKv;
	std::vector<int>* loWinIdx;
	std::vector<int>* hiWinIdx;
	std::vector<int>* devSeqLengthsDqdo;
	std::vector<int>* devSeqLengthsDkdv;
	float weightDecay_;
};