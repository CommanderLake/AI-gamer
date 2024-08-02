#pragma once
#include "common.h"
#include "Layer.h"
#include "Viewer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
#include <atomic>
class NeuralNetwork{
public:
	NeuralNetwork();
	~NeuralNetwork();
	void Initialize(int w, int h, bool train);
	__half* Forward(__half* data, bool train);
	__half* Backward(const __half* d_predictions, const float* d_targets);
	void UpdateParams();
	void SaveModel(const std::string& filename);
	void SaveOptimizerState(const std::string& filename);
	void Train(size_t count, int width, int height, Viewer* viewer);
	void Infer();
	void ListenForKey();
	static void ProcessOutput(const float* predictions);
	std::atomic<bool> simInput = false;
	std::atomic<bool> stopInfer = false;
private:
	cudnnHandle_t cudnn_;
	cublasHandle_t cublas_;
	std::vector<Layer*> layers_;
	__half* gradient_ = nullptr;
	int batchSize_;
	int seqLength_;
	int ctrlBatchSize_ = numCtrls_*batchSize_;
	float* ctrlBatchFloat_ = nullptr;
	__half* ctrlBatchHalf_ = nullptr;
	int inWidth_, inHeight_;
	float learningRate_;
	size_t maxBufferSize_;
};