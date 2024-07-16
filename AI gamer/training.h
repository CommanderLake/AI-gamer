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
	void initialize(int w, int h);
	__half* forward(__half* data, bool train);
	void backward(const __half* d_predictions, const float* d_targets);
	void updateParams();
	void train(InputRecord** data, size_t count);
private:
	cudnnHandle_t cudnn;
	cublasHandle_t cublas;
	std::vector<Layer*> layers;
	__half* gradient_ = nullptr;
	const size_t batchSize = 64;
	const int numCtrls = 16;
	const int ctrlBatchSize = numCtrls*batchSize;
	float* ctrlBatchFloat = nullptr;
	__half* ctrlBatchHalf = nullptr;
	int inWidth, inHeight;
	float learningRate = 0.0001f;
};