#pragma once
#include <cudnn.h>
struct __half;
class ConvScale{
public:
	explicit ConvScale(cudnnHandle_t cudnnHandle, int scaleFactor, int inWidth, int inHeight);
	~ConvScale();
	void ScaleInPlace(unsigned char* inImage);
	__half* ScaleToHalf(unsigned char* inImage);
	cudnnHandle_t cudnnHandle_;
	cudnnTensorDescriptor_t inDesc_, outDesc_;
	cudnnFilterDescriptor_t filterDesc_;
	cudnnConvolutionDescriptor_t convDesc_;
	void* dFilter_;
	size_t workspaceBytes_;
	void* dWorkspace_;
	__half* dIn_;
	__half* dOut_;
	int inHWC_, outHWC_;
	int inWidth_, inHeight_;
	int outWidth_, outHeight_;
	int scaleFactor_;
};