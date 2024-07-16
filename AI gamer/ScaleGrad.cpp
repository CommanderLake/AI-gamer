#include "ScaleGrad.h"
#include "common.h"
ScaleGrad::ScaleGrad(cudnnTensorDescriptor_t outDesc, float scale): scale_(scale){
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outSize_ = n*c*h*w;
}
ScaleGrad::~ScaleGrad(){
}
__half* ScaleGrad::forward(__half* data){
	return data;
}
__half* ScaleGrad::backward(__half* grad){
	scale(grad, outSize_, scale_);
	return grad;
}