#pragma once
#include "common.h"
#include "Activate.h"
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
	void Backward(const __half* d_predictions, const float* d_targets);
	void UpdateParams();
	void SaveModel(const std::string& filename);
	void SaveOptimizerState(const std::string& filename);
	void Train(InputRecord** data, size_t count);
	void Infer();
	void ListenForKey();
	static void ProcessOutput(const float* output);
	std::atomic<bool> simInput = false;
	std::atomic<bool> stopInfer = false;
private:
	cudnnHandle_t cudnn_;
	cublasHandle_t cublas_;
	std::vector<Layer*> layers_;
	__half* gradient_ = nullptr;
	size_t batchSize_;
	int ctrlBatchSize_ = numCtrls_*batchSize_;
	float* ctrlBatchFloat_ = nullptr;
	__half* ctrlBatchHalf_ = nullptr;
	int inWidth_, inHeight_;
	float learningRate_;
	size_t maxBufferSize_;
};