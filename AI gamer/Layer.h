#pragma once
#include <cuda_fp16.h>
#include <cudnn_ops_infer.h>
class Layer{
public:
	virtual ~Layer() = default;
	virtual __half* forward(__half* inData){ return nullptr; }
	virtual __half* backward(__half* inGrad){ return nullptr; }
	virtual void updateParameters(float learningRate){ }
	cudnnTensorDescriptor_t outDesc_;
	int outSize_;
	const char* layerName_;
};