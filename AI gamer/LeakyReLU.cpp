#include "LeakyReLU.h"
#include "common.h"
LeakyReLU::LeakyReLU(cudnnTensorDescriptor_t outDesc){
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outSize_ = n*c*h*w;
}
LeakyReLU::~LeakyReLU(){
}
__half* LeakyReLU::forward(__half* data){
	data_ = data;
	leakyRelu(data, outSize_, 0.01f);
	//printDataHalf(data_, 10, "LeakyReLU out");
	return data;
}
__half* LeakyReLU::backward(__half* grad){
	leakyReluBackward(grad, data_, grad, outSize_, 0.01f);
	return grad;
}