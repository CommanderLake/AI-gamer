#include "Sigmoid.h"
#include "common.h"
Sigmoid::Sigmoid(cudnnTensorDescriptor_t outDesc, int numSigmoidOutputs, int batchSize, const char* layerName) : numSigmoidOutputs_(numSigmoidOutputs), batchSize_(batchSize), data_(nullptr){
	layerName_ = layerName;
	outDesc_ = outDesc;
	cudnnDataType_t dt;
	int n, c, h, w, ns, cs, hs, ws;
	cudnnGetTensor4dDescriptor(outDesc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
	outCHW_ = c*h*w;
	outNCHW_ = n*outCHW_;
}
Sigmoid::~Sigmoid() {
}
__half* Sigmoid::Forward(__half* data) {
	data_ = data;
    SigmoidForward(data, numSigmoidOutputs_, batchSize_, outCHW_);
    return data;
}
__half* Sigmoid::Backward(__half* grad) {
    SigmoidBackward(grad, data_, numSigmoidOutputs_, batchSize_, outCHW_);
    return grad;
}