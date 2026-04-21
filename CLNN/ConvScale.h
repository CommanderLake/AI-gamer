#pragma once
#include <cudnn.h>
struct __half;
class __declspec(dllexport) ConvScale{
public:
	explicit ConvScale(cudnnHandle_t cudnnHandle, int filterSize, int stride, int padding, int batchSize, int channels, int* height, int* width);
	~ConvScale();
	void Forward();
	void ScaleUInt8InPlaceHost(unsigned char* inImage);
	void ScaleUInt8InPlaceDevice(unsigned char* inImage);
	__half* ScaleUInt8ToFP16Device(const unsigned char* inImage);
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t inDesc_, outDesc_;
	cudnnFilterDescriptor_t filterDesc_;
	cudnnConvolutionDescriptor_t convDesc_;
	cudnnConvolutionFwdAlgoPerf_t algoPerf_;
	void* dFilter_;
	size_t workspaceBytes_;
	void* dWorkspace_;
	unsigned char* dInB_;
	__half* dIn_;
	__half* dOut_;
	int batchSize_;
	int inC_;
	int inNCHW_, outNCHW_;
	int inWidth_, inHeight_;
	int outWidth_, outHeight_;
	int stride_, filterSize_, padding_;
	float alpha_ = 1.0f, beta_ = 0.0f;
};