#pragma once
#include <cuda_fp16.h>
#include <cudnn_ops_infer.h>
#include <fstream>
class Layer{
public:
	virtual ~Layer() = default;
	virtual __half* Forward(__half* inData, bool train){ return nullptr; }
	virtual __half* Backward(__half* inGrad){ return nullptr; }
	virtual void UpdateParameters(float learningRate){}
	virtual void SaveParameters(std::ofstream& file, float* buffer){}
	virtual void LoadParameters(std::ifstream& file, float* buffer){}
	virtual void SaveOptimizerState(std::ofstream& file, float* buffer){}
	virtual void LoadOptimizerState(std::ifstream& file, float* buffer){}
	virtual bool HasParameters(){ return false; }
	virtual bool HasOptimizerState(){ return false; }
	virtual size_t GetParameterSize(){ return 0; }
	virtual size_t GetOptimizerStateSize(){ return 0; }
	cudnnTensorDescriptor_t outDesc_;
	int outNCHW_;
	const char* layerName_ = "";
	bool train_;
	size_t weightCount_ = 0;
};