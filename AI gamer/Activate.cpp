#include "Activate.h"
#include "common.h"
Activate::Activate(cudnnHandle_t cudnnHandle, cudnnTensorDescriptor_t outDesc, cudnnActivationMode_t mode, double coef): cudnnHandle_(cudnnHandle){
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outSize_ = n*c*h*w;
	checkCUDNN(cudnnCreateActivationDescriptor(&activDesc_));
	checkCUDNN(cudnnSetActivationDescriptor(activDesc_, mode, CUDNN_NOT_PROPAGATE_NAN, coef));
}
Activate::~Activate(){
	cudnnDestroyActivationDescriptor(activDesc_);
}
__half* Activate::forward(__half* data){
	data_ = data;
	checkCUDNN(cudnnActivationForward(cudnnHandle_, activDesc_, &alpha, outDesc_, data, &beta0, outDesc_, data));
	return data;
}
__half* Activate::backward(__half* grad){
	checkCUDNN(cudnnActivationBackward(cudnnHandle_, activDesc_, &alpha, outDesc_, data_, outDesc_, grad, outDesc_, data_, &beta1, outDesc_, grad));
	return grad;
}