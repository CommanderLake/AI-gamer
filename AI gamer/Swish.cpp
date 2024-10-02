#include "Swish.h"
#include "common.h"
Swish::Swish(int size, const char* layerName){
	layerName_ = layerName;
	outNCHW_ = size;
}
Swish::~Swish(){
}
__half* Swish::Forward(__half* data){
	data_ = data;
	SwishForward(data, outNCHW_);
	return data;
}
__half* Swish::Backward(__half* grad){
	SwishBackward(grad, data_, outNCHW_);
	return grad;
}