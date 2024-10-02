#pragma once
#include "common.h"
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
class NN{
public:
	NN(cudnnHandle_t cudnnHandle, cublasHandle_t cublasHandle, int w, int h, bool train);
	~NN();
	__half* Forward(__half* data);
	__half* Backward(__half* grad);
	void UpdateParams();
	void SaveModel(const std::string& filename);
	void SaveOptimizerState(const std::string& filename);
	cudnnHandle_t cudnn_;
	cublasHandle_t cublas_;
	std::vector<Layer*> layers_;
	int batchSize_;
	int seqLength_;
	int inWidth_, inHeight_;
	float learningRate_;
	size_t maxBufferSize_;
};