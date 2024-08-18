#pragma once
#include "common.h"
#include "Layer.h"
#include <cudnn.h>
#include <cublas_v2.h>
#include <vector>
class NN{
public:
	NN(int w, int h, bool train);
	~NN();
	__half* Forward(__half* data);
	__half* Backward(__half* grad);
	void UpdateParams();
	void SaveModel(const std::string& filename);
	void SaveOptimizerState(const std::string& filename);
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