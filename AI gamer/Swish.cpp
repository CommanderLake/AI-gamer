#include "Swish.h"
#include "common.h"
Swish::Swish(int size, const char* layerName){
	layerName_ = layerName;
	outNCHW_ = size;
	CUDAMallocZero(&data_, outNCHW_);
}
Swish::~Swish(){
	cudaFree(data_);
}
__half* Swish::Forward(__half* data){
	SwishForward(data, data_, outNCHW_);
	return data_;
}
__half* Swish::Backward(__half* grad){
	SwishBackward(grad, data_, outNCHW_);
	return grad;
}