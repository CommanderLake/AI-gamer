#pragma once
#include "common.h"
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
class Discriminator{
public:
	Discriminator(int batchSize, int inputSize, bool train);
	~Discriminator();
	__half* Forward(__half* data);
	__half* Backward(const __half* predictions, const __half* targets);
	void UpdateParams();
	void SaveModel(const std::string& filename);
	void SaveOptimizerState(const std::string& filename);
	void LoadModel(const std::string& filename);
	void LoadOptimizerState(const std::string& filename);
	cudnnHandle_t cudnn_;
	cublasHandle_t cublas_;
	std::vector<Layer*> layers_;
	__half* gradient_;
	int inputSize_;
	int batchSize_;
	float learningRate_;
	size_t maxBufferSize_;
};