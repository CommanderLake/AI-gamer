#include "GELULayer.h"
#include "common.h"
GELU::GELU(const int batchSize, const int channels, const int height, const int width, const char* layerName) : batchSize_(batchSize), outC_(channels), outHeight_(height), outWidth_(width){
	layerName_ = layerName;
	outNCHW_ = batchSize_*outC_*outHeight_*outWidth_;
	CUDAMallocZero(&dataOut_, outNCHW_);
}
GELU::~GELU(){
	cudaFree(dataOut_);
}
__half* GELU::Forward(__half* data){
	data_ = data;
	GELUForward(data, dataOut_, outNCHW_, nullptr);
	return dataOut_;
}
__half* GELU::Backward(__half* grad){
	GELUBackward(grad, data_, outNCHW_, nullptr);
	return grad;
}
void GELU::SetTrain(const bool enable){
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