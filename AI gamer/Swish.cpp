#include "Swish.h"
#include "common.h"
Swish::Swish(int batchSize, int channels, int height, int width, const char* layerName): batchSize_(batchSize), outC_(channels), outHeight_(height), outWidth_(width){
	layerName_ = layerName;
	outNCHW_ = batchSize_*outC_*outHeight_*outWidth_;
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
void Swish::SetTrain(bool enable){
	int bs;
	if(enable){
		train_ = true;
		bs = batchSize_;
	} else{
		train_ = false;
		bs = 1;
	}
	outNCHW_ = bs*outC_*outHeight_*outWidth_;
}