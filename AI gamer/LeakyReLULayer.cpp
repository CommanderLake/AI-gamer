#include "LeakyReLULayer.h"
#include "common.h"
LeakyReLU::LeakyReLU(const int batchSize, const int channels, const int height, const int width, const char* layerName): batchSize_(batchSize), outC_(channels), outHeight_(height), outWidth_(width), slope_(1.0f/128.0f){
	layerName_ = layerName;
	outNCHW_ = batchSize_*outC_*outHeight_*outWidth_;
	CUDAMallocZero(&dataOut_, outNCHW_);
}
LeakyReLU::~LeakyReLU(){
	cudaFree(dataOut_);
}
__half* LeakyReLU::Forward(__half* data){
	data_ = data;
	LeakyReluForward(data, dataOut_, outNCHW_, slope_, nullptr);
	return dataOut_;
}
__half* LeakyReLU::Backward(__half* grad){
	LeakyReluBackward(grad, data_, outNCHW_, slope_, nullptr);
	return grad;
}
void LeakyReLU::SetTrain(const bool enable){
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