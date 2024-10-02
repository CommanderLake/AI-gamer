#include "Sigmoid.h"
#include "common.h"
Sigmoid::Sigmoid(int numSigmoidOutputs, int batchSize, int outC, const char* layerName) : numSigmoidOutputs_(numSigmoidOutputs), outC_(outC), batchSize_(batchSize), data_(nullptr){
	layerName_ = layerName;
	outNCHW_ = batchSize_*outC_;
}
Sigmoid::~Sigmoid() {}
__half* Sigmoid::Forward(__half* data) {
	data_ = data;
	SigmoidForward(data, outC_, numSigmoidOutputs_, outNCHW_);
    return data;
}
__half* Sigmoid::Backward(__half* grad) {
	SigmoidBackward(grad, data_, outC_, numSigmoidOutputs_, outNCHW_);
    return grad;
}