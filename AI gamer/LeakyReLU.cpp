#include "LeakyReLU.h"
#include "common.h"
LeakyReLU::LeakyReLU(cudnnTensorDescriptor_t outDesc, const char* layerName): slope_(1.0f/128.0f){
	layerName_ = layerName;
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outNCHW_ = n*c*h*w;
}
LeakyReLU::~LeakyReLU(){
}
__half* LeakyReLU::Forward(__half* data, bool train){
	data_ = data;
	LeakyRelu(data, outNCHW_, slope_);
	return data;
}
__half* LeakyReLU::Backward(__half* grad){
	LeakyReluBackward(grad, data_, outNCHW_, slope_);
	return grad;
}