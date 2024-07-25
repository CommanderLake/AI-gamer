#include "ScaleGrad.h"
#include "common.h"
ScaleGrad::ScaleGrad(cudnnTensorDescriptor_t outDesc, float scale): scale_(scale){
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outNCHW_ = n*c*h*w;
}
ScaleGrad::~ScaleGrad(){
}
__half* ScaleGrad::Forward(__half* data, bool train){
	return data;
}
__half* ScaleGrad::Backward(__half* grad){
	Scale(grad, outNCHW_, scale_);
	return grad;
}