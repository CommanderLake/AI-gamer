#include "LeakyReLU.h"
#include "common.h"
LeakyReLU::LeakyReLU(int batchSize, int channels, int height, int width, const char* layerName): batchSize_(batchSize), outC_(channels), outHeight_(height), outWidth_(width), slope_(1.0f/128.0f){
	layerName_ = layerName;
	outNCHW_ = batchSize_*outC_*outHeight_*outWidth_;
	CUDAMallocZero(&data_, outNCHW_);
}
LeakyReLU::~LeakyReLU(){
	cudaFree(data_);
}
__half* LeakyReLU::Forward(__half* data){
	LeakyReluForward(data, outNCHW_, slope_);
	return data;
}
__half* LeakyReLU::Backward(__half* grad){
	LeakyReluBackward(grad, data_, outNCHW_, slope_);
	return grad;
}
void LeakyReLU::SetTrain(bool enable){
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