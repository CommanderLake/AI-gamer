#include "SigmoidLayer.h"
#include "common.h"
SigmoidLayer::SigmoidLayer(const int numSigmoidOutputs, const int batchSize, const int outC, const char* layerName) : numSigmoidOutputs_(numSigmoidOutputs), outC_(outC), batchSize_(batchSize), data_(nullptr){
	layerName_ = layerName;
	outNCHW_ = batchSize_*outC_;
}
SigmoidLayer::~SigmoidLayer() {}
__half* SigmoidLayer::Forward(__half* data) {
	data_ = data;
	SigmoidForward(data, outC_, numSigmoidOutputs_, outNCHW_);
    return data;
}
__half* SigmoidLayer::Backward(__half* grad) {
	SigmoidBackward(grad, data_, outC_, numSigmoidOutputs_, outNCHW_);
    return grad;
}