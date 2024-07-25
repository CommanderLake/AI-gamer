#pragma once
#include <cuda_fp16.h>
#include <cudnn_ops_infer.h>
#include <fstream>
class Layer{
public:
	virtual ~Layer() = default;
	virtual __half* Forward(__half* inData, bool train){ return nullptr; }
	virtual __half* Backward(__half* inGrad){ return nullptr; }
	virtual void UpdateParameters(float learningRate){ }
	virtual void SaveParameters(std::ofstream& file, float* buffer) const{}
	virtual void LoadParameters(std::ifstream& file, float* buffer){}
	virtual void SaveOptimizerState(std::ofstream& file, float* buffer) const{}
	virtual void LoadOptimizerState(std::ifstream& file, float* buffer){}
	virtual bool HasParameters() const{ return false; }
	virtual bool HasOptimizerState() const{ return false; }
	virtual size_t GetParameterSize() const{ return 0; }
	virtual size_t GetOptimizerStateSize() const{ return 0; }
	cudnnTensorDescriptor_t outDesc_;
	int outNCHW_;
	const char* layerName_ = "";
	bool train_;
};