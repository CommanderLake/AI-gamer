#pragma once
#include "Layer.h"
#include <cudnn.h>

class __declspec(dllexport) Conv1DLayer final : public Layer{
public:
	const bool useAdamW_ = true;
	struct ConvolutionAlgorithms{
		cudnnConvolutionFwdAlgo_t fwdAlgo;
		cudnnConvolutionBwdDataAlgo_t bwdDataAlgo;
		cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo;
		size_t workspaceSize;
	};
	Conv1DLayer(cudnnHandle_t cudnnHandle, int batchSize, int inputChannels, int outputChannels, int kernelSize, int stride, int dilation, int* width, std::string layerName, bool train, float weightDecay, int gradAccumLength, WeightInitMethod weightInitMethod, bool backpropInput = true);
	~Conv1DLayer() override;
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
	void CollectAdamWTasks(std::vector<AdamWHalfTask>& halfTasks, std::vector<AdamWFloatTask>& floatTasks) override;
	size_t WorkspaceSize() const{ return algos_.workspaceSize; }
	void ReleaseWorkspace();
	void UseSharedWorkspace(void* workspace, size_t workspaceSize);
	static ConvolutionAlgorithms GetConvolutionAlgorithms(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, bool isTraining, bool backpropInput = true);
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t inDesc_, outDesc_;
	cudnnFilterDescriptor_t filterDesc_;
	cudnnConvolutionDescriptor_t convDesc_;
	ConvolutionAlgorithms algos_;
	int batchSize_, inC_, outC_, inWidth_, outWidth_, kernelSize_, stride_, dilation_;
	int inNCHW_;
	__half* inData_ = nullptr;
	__half* outData_ = nullptr;
	__half* outGrad_ = nullptr;
	__half* gradWeights_ = nullptr;
	void* workspace_ = nullptr;
	__half *m_Weights_ = nullptr, *v_Weights_ = nullptr;
	int t_ = 0;
	float alphaWeights_ = 1.0f;
	const float zero_ = 0.0f;
	const float one_ = 1.0f;
	float weightDecay_;
	int gradAccumLength_;
	int accumCount_ = 0;
	bool backpropInput_;
	bool ownsWorkspace_ = false;
};
