#include "SigmoidLayer.h"
#include "CuCommon.h"
SigmoidLayer::SigmoidLayer(const int batchSize, const int numCtrls, const int numButs, std::string layerName) : batchSize_(batchSize), numCtrls_(numCtrls), numButs_(numButs){
	layerName_ = layerName;
	outNCHW_ = batchSize_*numCtrls_;
}
SigmoidLayer::~SigmoidLayer(){
}
__half* SigmoidLayer::Forward(__half* data) {
	data_ = data;
	SigmoidForward(data, data, numCtrls_, numButs_, outNCHW_);
    return data;
}
__half* SigmoidLayer::Backward(__half* grad) {
	SigmoidBackward(grad, data_, numCtrls_, numButs_, outNCHW_);
    return grad;
}