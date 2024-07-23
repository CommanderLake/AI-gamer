#pragma once
#include "common.h"
#include "Activate.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
class NeuralNetwork{
public:
	NeuralNetwork();
	~NeuralNetwork();
	void Initialize(int w, int h);
	__half* Forward(__half* data, bool train);
	void Backward(const __half* d_predictions, const float* d_targets);
	void UpdateParams();
	void Train(InputRecord** data, size_t count);
private:
	cudnnHandle_t cudnn_;
	cublasHandle_t cublas_;
	std::vector<Layer*> layers_;
	__half* gradient_ = nullptr;
	const size_t batchSize_;
	const int ctrlBatchSize_ = numCtrls_*batchSize_;
	float* ctrlBatchFloat_ = nullptr;
	__half* ctrlBatchHalf_ = nullptr;
	int inWidth_, inHeight_;
	float learningRate_;
};