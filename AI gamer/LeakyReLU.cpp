#include "LeakyReLU.h"
#include "common.h"
LeakyReLU::LeakyReLU(int size, const char* layerName): slope_(1.0f/128.0f){
	layerName_ = layerName;
	outNCHW_ = size;
}
LeakyReLU::~LeakyReLU(){
}
__half* LeakyReLU::Forward(__half* data){
	data_ = data;
	LeakyReluForward(data, outNCHW_, slope_);
	return data;
}
__half* LeakyReLU::Backward(__half* grad){
	LeakyReluBackward(grad, data_, outNCHW_, slope_);
	return grad;
}