#pragma once
#include <cuda_fp16.h>
#include <cudnn_ops_infer.h>
class Layer{
public:
	virtual ~Layer() = default;
	virtual __half* Forward(__half* inData){ return nullptr; }
	virtual __half* Backward(__half* inGrad){ return nullptr; }
	virtual void UpdateParameters(float learningRate){ }
	cudnnTensorDescriptor_t outDesc_;
	int outNCHW_;
	const char* layerName_ = "";
};